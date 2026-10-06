"""账户退付：把应退凭证形成的账户信用按退付单实际支付给调用方。

口径（完全基于 app/ledger.py 落盘的已记凭证、app/payments.py 落盘的收款
核销与本模块落盘的退付，凭证金额一律取入账时冻结的计费栏，不重新核算）：
1. 调用方按账户提交全平台唯一退付单号 refund_no 与**正金额**；
2. 可退付上限为账户**当前信用**：已记应退凭证按原入账顺序抵减尚未付清的
   正额凭证、并扣除已落盘退付后剩余的应退净额（credit_yuan，见
   app/payments.account_position）；退付金额**超过当前信用整笔拒绝**：
   返回 422、不写退付主行（不留任何记录）；金额必须为正；
3. 退付成功后，台账查询与收款查询的当前信用立即扣除已付退款：重算头寸时
   已付退款先从应退池划走（冻结事实，与现金核销受保护同构），剩余池子再
   按原规则 FIFO 抵减正额凭证——后续入账的应补/应退仍按原规则改变最新
   尚欠与信用，而已记凭证、收款分配与首次回执保持原样；
4. 返回本次**退付金额、退付时间与退付后的头寸**（最新尚欠、剩余信用、
   累计已付退款、账户退付笔数），并提供按退付次序查询退付记录的 API；
5. **幂等**：同一 refund_no + 同账户 + 同金额重试，原样返回**首次回执**
   （result=deduped，金额、时间与退付后头寸快照均为首次落盘值，不随后来
   新入账的应补/应退凭证或新收款漂移）；同一单号换了金额、或拿到别的
   账户使用 → **409 冲突**（refund_no_conflict），不写入；
6. 信用核对、退付登记与落盘和原凭证入账、收款**共用同一把 SQLite 写事务
   锁**（RLock + BEGIN IMMEDIATE）：refund_no 主键 + (账户, 序号) 唯一键
   在数据库层兜底，并发提交同一单号只有一笔落盘、并发不同单号串行核对
   信用不会重复支付，进程重启后退付记录仍在、不会重复退付；信用不足/
   冲突一律回滚，不留部分数据。

HTTP 错误码：
- refund_amount_not_positive：退付金额非正（422，正常由请求体校验拦截，
  服务层兜底亚微级金额）；
- refund_exceeds_credit：退付金额超过当前信用，整笔拒绝、不留记录
  （422，错误体附 outstanding_yuan / credit_yuan）；
- refund_no_conflict：退付单号已使用但金额或账户不一致（409，错误体
  附已登记的 account_id / amount_yuan）。
"""

from __future__ import annotations

import json
import uuid
from typing import List, Optional

from app.payments import _EPS, _round6, _utc_now_iso, account_position
from app.storage import SessionStore, get_store


class RefundError(Exception):
    """退付操作失败：code 决定 HTTP 错误体 error 字段，status_code 决定状态码。"""

    def __init__(
        self,
        code: str,
        reason: str,
        status_code: int = 422,
        extra: Optional[dict] = None,
    ):
        super().__init__(reason)
        self.code = code
        self.status_code = status_code
        # 附加到错误响应体顶层的结构化字段（如冲突单号的账户/金额、当前信用）
        self.extra = extra or {}


def submit_refund(
    account_id: str,
    refund_no: str,
    amount_yuan: float,
    store: Optional[SessionStore] = None,
) -> dict:
    """按账户登记一笔退付：核对当前信用后原子落盘，信用不足整笔拒绝。

    同单号同账户同金额重试返回首付回执；同单号换金额或跨账户 → 409。
    """
    store = store or get_store()
    amount_yuan = _round6(amount_yuan)
    # 请求模型已要求 gt=0；这里防住亚微级金额（如 1e-12 被 6 位舍入为 0）：
    # 归一后非正视为非法，不允许「0 元退付」生成空记录。
    if amount_yuan <= 0.0:
        raise RefundError(
            "refund_amount_not_positive",
            "退付金额必须为正且至少达到 0.000001 元",
            422,
            extra={"refund_no": refund_no, "amount_yuan": amount_yuan},
        )

    with store.write_transaction() as conn:
        # 1) 退付单号全平台唯一：先查重，区分「原样重试」与「单号冲突」。
        existing = store.get_refund_payout(conn, refund_no)
        if existing is not None:
            same_account = existing["account_id"] == account_id
            same_amount = abs(float(existing["amount_yuan"]) - amount_yuan) < _EPS
            if same_account and same_amount:
                # 原样重试：冻结回执原样取回，仅 result 标记 deduped
                # （即使后来又入账了新的应补/应退凭证，回执也不漂移）。
                receipt = json.loads(existing["receipt_json"])
                receipt["result"] = "deduped"
                return receipt
            raise RefundError(
                "refund_no_conflict",
                f"退付单号 {refund_no} 已登记到账户 {existing['account_id']}、"
                f"金额 {float(existing['amount_yuan'])} 元；同一单号不能换金额"
                "或跨账户使用，本次退付整笔拒绝、不留记录",
                409,
                extra={
                    "refund_no": refund_no,
                    "account_id": existing["account_id"],
                    "amount_yuan": float(existing["amount_yuan"]),
                },
            )

        # 2) 重算当前头寸（应退抵减并扣除已付退款后的信用）。
        position = account_position(conn, store, account_id)
        totals = position["totals"]
        outstanding = totals["outstanding_yuan"]
        credit = totals["credit_yuan"]

        # 3) 超过当前信用整笔拒绝（无信用 / 信用不足都在此拒绝，不留记录）。
        #    金额与头寸均经 _round6 归一，亚微级浮点残差先归零再比较，
        #    业务金额（6 位小数）边界精确：等于信用可退，大于信用整笔拒绝。
        if amount_yuan > credit:
            raise RefundError(
                "refund_exceeds_credit",
                f"退付 {amount_yuan} 元超过账户 {account_id} 当前信用 "
                f"{credit} 元（当前尚欠 {outstanding} 元）；超过当前信用的退付"
                "整笔拒绝，不登记退付单、不留记录",
                422,
                extra={
                    "refund_no": refund_no,
                    "amount_yuan": amount_yuan,
                    "outstanding_yuan": outstanding,
                    "credit_yuan": credit,
                },
            )

        # 4) 组装首次回执（落盘冻结）：退付金额、时间与退付后的头寸快照。
        paid_at = _utc_now_iso()
        refund_count = store.count_refund_payouts(conn, account_id) + 1
        receipt = {
            "account_id": account_id,
            "refund_no": refund_no,
            "result": "paid",
            "amount_yuan": amount_yuan,
            "paid_at": paid_at,
            "outstanding_yuan": outstanding,
            "credit_yuan": _round6(credit - amount_yuan),
            "total_refunded_yuan": _round6(
                totals["total_refunded_yuan"] + amount_yuan
            ),
            "refund_count": refund_count,
        }

        # 5) 退付主行（含冻结回执 JSON）落盘；refund_no 主键兜底，
        #    并发同单号只有一笔成功，信用核对与登记在同一事务内不会重复支付。
        store.insert_refund_payout(
            conn,
            payout_id=str(uuid.uuid4()),
            refund_no=refund_no,
            account_id=account_id,
            refund_seq=refund_count,
            amount_yuan=amount_yuan,
            receipt_json=json.dumps(receipt, ensure_ascii=False),
            paid_at=paid_at,
        )
        return receipt


def list_refunds(
    account_id: str,
    store: Optional[SessionStore] = None,
) -> dict:
    """按退付次序查询账户全部退付记录（均为落盘时冻结的首付回执）。"""
    store = store or get_store()
    with store._connect() as conn:
        position = account_position(conn, store, account_id)
        totals = position["totals"]
        rows = store.list_refund_payout_rows(account_id)

    refunds: List[dict] = []
    total_refunded_check = 0.0
    for row in rows:
        receipt = json.loads(row["receipt_json"])
        # 正常列表查询：退付记录保持 paid（重试时才在提交接口标 deduped）。
        receipt["result"] = "paid"
        refunds.append(receipt)
        total_refunded_check = _round6(
            total_refunded_check + float(row["amount_yuan"])
        )

    return {
        "account_id": account_id,
        "refund_count": len(refunds),
        "total_refunded_yuan": _round6(total_refunded_check),
        "outstanding_yuan": totals["outstanding_yuan"],
        "credit_yuan": totals["credit_yuan"],
        "refunds": refunds,
    }
