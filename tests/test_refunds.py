"""账户退付行为测试（应退抵减后的当前信用 → 实际退付）。

覆盖：
- 端到端：收款付清后计量表更正产生应退凭证形成信用，退付从信用中实际支付；
  退付后台账与收款查询的当前信用立即扣除已付退款（total_refunded_yuan 同步展示）；
- 只能支付当前信用：超过信用（含无信用、空账户）整笔拒绝且不留任何记录；
  恰好等于信用可以（边界）；
- 同一单号 + 同账户 + 同金额重试返回首次冻结回执（result=deduped），即使后来
  入账新的应补/应退凭证，回执（金额、时间、退付后头寸快照）不漂移；
- 同一单号换金额、跨账户使用 → 409 refund_no_conflict，附已登记账户/金额；
- 后续入账的应补/应退仍按原规则改变最新尚欠与信用：已付退款先从应退池划走，
  之后入账的应补形成尚欠、应退重新形成信用并可再次退付；
- 退付不改写已记凭证、收款分配与首次回执（旧收款回执与逐凭证核销冻结）；
- 退付记录按退付次序查询、内容为首付冻结值、账户级头寸为最新值；
- 非正金额 / 亚微级金额 / 空或超长单号拦截；
- 并发：同一单号多线程提交只有一笔落盘（其余 deduped）；不同单号并发退付
  串行核对信用，累计退付不超过信用、不重复支付；
- 按同一路径重建存储模拟进程重启：退付记录、幂等重试与信用核对仍正确；
- 旧库（无退付表）打开自动补建；既有收款/入账/出单等接口行为不变。
"""

import threading
import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.refunds import submit_refund
from app.storage import configure_store, get_store

client = TestClient(app)


DISPATCH = dict(
    prices=[0.0, 1.0, 1.0, 0.0],
    base_load_kw=[0.2, 0.2, 0.2, 0.2],
    pv_kw=[0.0, 0.0, 0.0, 0.0],
    capacity_kwh=0.4,
    max_charge_kw=1.0,
    max_discharge_kw=1.0,
    charge_efficiency=0.9,
    discharge_efficiency=0.9,
    initial_soc_kwh=0.0,
    min_terminal_soc_kwh=0.0,
    degradation_cost_per_kwh=0.2,
)

RECORDS = [
    dict(
        base_load_kw=0.2, pv_kw=0.0,
        charge_ac_energy_kwh=round(0.2 / 0.9, 6), discharge_ac_energy_kwh=0.0,
        grid_purchase_energy_kwh=round(0.1 + 0.2 / 0.9, 6),
        soc_start_kwh=0.0, soc_end_kwh=0.2,
    ),
    dict(
        base_load_kw=0.2, pv_kw=0.0,
        charge_ac_energy_kwh=0.0, discharge_ac_energy_kwh=0.09,
        grid_purchase_energy_kwh=0.01,
        soc_start_kwh=0.2, soc_end_kwh=0.1,
    ),
    dict(
        base_load_kw=0.2, pv_kw=0.0,
        charge_ac_energy_kwh=0.0, discharge_ac_energy_kwh=0.09,
        grid_purchase_energy_kwh=0.01,
        soc_start_kwh=0.1, soc_end_kwh=0.0,
    ),
    dict(
        base_load_kw=0.2, pv_kw=0.0,
        charge_ac_energy_kwh=0.0, discharge_ac_energy_kwh=0.0,
        grid_purchase_energy_kwh=0.1,
        soc_start_kwh=0.0, soc_end_kwh=0.0,
    ),
]

TOL = 0.01


@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    configure_store(str(tmp_path / "refunds.db"))
    yield
    configure_store(None)


def _plan():
    return client.post("/dispatch", json=DISPATCH).json()


def _create_complete():
    sid = client.post(
        "/dispatch/sessions",
        json={"dispatch": DISPATCH, "plan_snapshot": _plan(), "audit_tolerance_kwh": TOL},
    ).json()["session_id"]
    assert client.post(
        f"/dispatch/sessions/{sid}/batches",
        json={"start_index": 0, "records": RECORDS[:2]},
    ).status_code == 200
    assert client.post(
        f"/dispatch/sessions/{sid}/batches",
        json={"start_index": 2, "records": RECORDS[2:]},
    ).status_code == 200
    return sid


def _rev(sid, expected, changes):
    return client.post(
        f"/dispatch/sessions/{sid}/revisions",
        json={"expected_revision": expected, "changes": changes},
    )


def _issue(sid, revision):
    return client.post(
        f"/dispatch/sessions/{sid}/vouchers",
        json={"expected_revision": revision},
    )


def _ch(index, record):
    return {"index": index, "record": record}


def _post_vouchers(account, items):
    return client.post(
        f"/accounts/{account}/ledger",
        json={"items": [{"session_id": s, "voucher_no": n} for s, n in items]},
    )


def _pay(account, payment_no, amount):
    return client.post(
        f"/accounts/{account}/payments",
        json={"payment_no": payment_no, "amount_yuan": amount},
    )


def _refund(account, refund_no, amount):
    return client.post(
        f"/accounts/{account}/refunds",
        json={"refund_no": refund_no, "amount_yuan": amount},
    )


def _ledger(account):
    return client.get(f"/accounts/{account}/ledger").json()


def _payments(account):
    return client.get(f"/accounts/{account}/payments").json()


def _refunds(account):
    return client.get(f"/accounts/{account}/refunds").json()


def _raw_entry(account, entry_no, sid, vno, signed, billing):
    """直接落一条台账明细（绕过费用非负的业务约束，精确控制头寸口径）。"""
    store = get_store()
    with store.write_transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO sessions(session_id, dispatch_json,"
            " plan_snapshot_json, audit_tolerance_kwh, num_periods, created_at)"
            " VALUES (?,?,?,?,?,?)",
            (sid, "{}", "{}", TOL, 4, "t"),
        )
        conn.execute(
            "INSERT OR IGNORE INTO session_settlement_vouchers(voucher_id, session_id,"
            " voucher_no, revision_no, current_revision_at_issue, ledger_json,"
            " settlement_json, total_cost_yuan, billing_type, amount_yuan,"
            " relative_to_revision, relative_to_voucher_no, issued_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), sid, vno, 0, 0, "[]", "{}",
             abs(signed), billing, abs(signed), None, None, "t"),
        )
        conn.execute(
            "INSERT INTO account_ledger_entries(entry_id, account_id, entry_no,"
            " session_id, voucher_no, voucher_id, revision_no, billing_type,"
            " voucher_amount_yuan, relative_to_revision, relative_to_voucher_no,"
            " signed_amount_yuan, balance_after_yuan, total_cost_yuan, posted_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), account, entry_no, sid, vno, str(uuid.uuid4()), 0,
             billing, abs(signed), None, None, signed, signed, abs(signed), "t"),
        )


def _make_e2e_credit(account):
    """端到端造信用：full 0.10 入账并收清 → 两次更正各产生应退 0.01 → 信用 0.02。"""
    sid = _create_complete()
    _issue(sid, 0)  # 凭证 1：full 0.10
    _post_vouchers(account, [(sid, 1)])
    assert _pay(account, "PAY-FULL", 0.10).status_code == 200
    # 更正 t1：负荷 0.18、购电 0（总费用 0.10 → 0.09），凭证 2 应退 0.01
    down1 = dict(RECORDS[1])
    down1.update(base_load_kw=0.18, grid_purchase_energy_kwh=0.0)
    assert _rev(sid, 0, [_ch(1, down1)]).status_code == 200
    v2 = _issue(sid, 1).json()
    assert v2["billing"]["type"] == "refund"
    assert v2["billing"]["amount_yuan"] == pytest.approx(0.01, abs=1e-9)
    # 更正 t2：负荷 0.18、购电 0（总费用 0.09 → 0.08），凭证 3 应退 0.01
    down2 = dict(RECORDS[2])
    down2.update(base_load_kw=0.18, grid_purchase_energy_kwh=0.0)
    assert _rev(sid, 1, [_ch(2, down2)]).status_code == 200
    v3 = _issue(sid, 2).json()
    assert v3["billing"]["type"] == "refund"
    assert v3["billing"]["amount_yuan"] == pytest.approx(0.01, abs=1e-9)
    assert _post_vouchers(account, [(sid, 2), (sid, 3)]).status_code == 200
    return sid


# ---------------------------------------------------------------------------
# 基本退付：信用支付、台账/收款查询同步扣减、恰好退清
# ---------------------------------------------------------------------------


def test_refund_payout_from_credit_end_to_end():
    sid = _make_e2e_credit("fam")
    led = _ledger("fam")
    assert led["outstanding_yuan"] == 0.0
    assert led["credit_yuan"] == pytest.approx(0.02, abs=1e-9)
    assert led["total_refunded_yuan"] == 0.0

    # 退付 0.005：小于信用，成功；回执给退付金额、时间与退付后头寸
    r1 = _refund("fam", "RF-1", 0.005)
    assert r1.status_code == 200, r1.text
    b1 = r1.json()
    assert b1["result"] == "paid"
    assert b1["account_id"] == "fam"
    assert b1["refund_no"] == "RF-1"
    assert b1["amount_yuan"] == pytest.approx(0.005, abs=1e-9)
    assert b1["paid_at"].endswith("Z")
    assert b1["outstanding_yuan"] == 0.0
    assert b1["credit_yuan"] == pytest.approx(0.015, abs=1e-9)
    assert b1["total_refunded_yuan"] == pytest.approx(0.005, abs=1e-9)
    assert b1["refund_count"] == 1

    # 台账与收款查询的当前信用立即扣除已付退款
    led = _ledger("fam")
    assert led["credit_yuan"] == pytest.approx(0.015, abs=1e-9)
    assert led["total_refunded_yuan"] == pytest.approx(0.005, abs=1e-9)
    pays = _payments("fam")
    assert pays["credit_yuan"] == pytest.approx(0.015, abs=1e-9)
    assert pays["total_refunded_yuan"] == pytest.approx(0.005, abs=1e-9)
    # 收款记录本身不受退付影响
    assert pays["payment_count"] == 1
    assert pays["total_received_yuan"] == pytest.approx(0.10, abs=1e-9)

    # 恰好退清剩余信用（边界：等于信用可以）
    r2 = _refund("fam", "RF-2", 0.015)
    assert r2.status_code == 200, r2.text
    b2 = r2.json()
    assert b2["credit_yuan"] == 0.0
    assert b2["total_refunded_yuan"] == pytest.approx(0.02, abs=1e-9)
    assert b2["refund_count"] == 2

    assert _ledger("fam")["credit_yuan"] == 0.0
    assert _ledger("fam")["total_refunded_yuan"] == pytest.approx(0.02, abs=1e-9)
    # 信用已退清：再退任何正金额都整笔拒绝
    r3 = _refund("fam", "RF-3", 0.001)
    assert r3.status_code == 422
    assert r3.json()["error"] == "refund_exceeds_credit"
    assert _refunds("fam")["refund_count"] == 2
    # 已记凭证明细不被退付改写（冻结字段与逐凭证重算列保持）
    entries = _ledger("fam")["entries"]
    assert [e["billing"]["type"] for e in entries] == ["full", "refund", "refund"]
    assert entries[0]["paid_yuan"] == pytest.approx(0.10, abs=1e-9)
    assert all(e["outstanding_yuan"] == 0.0 for e in entries)


def test_refund_exceeds_credit_rejected_without_any_trace():
    _raw_entry("fam", 1, "s1", 1, 0.10, "full")
    _raw_entry("fam", 2, "s1", 2, -0.15, "refund")  # 抵 0.10 后信用 0.05
    assert _ledger("fam")["credit_yuan"] == pytest.approx(0.05, abs=1e-9)

    r = _refund("fam", "TOO-BIG", 0.06)  # 信用仅 0.05
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "refund_exceeds_credit"
    assert body["refund_no"] == "TOO-BIG"
    assert body["amount_yuan"] == pytest.approx(0.06, abs=1e-9)
    assert body["outstanding_yuan"] == 0.0
    assert body["credit_yuan"] == pytest.approx(0.05, abs=1e-9)

    # 整笔拒绝不留记录：退付列表为空、累计已付 0、信用不变
    lst = _refunds("fam")
    assert lst["refund_count"] == 0
    assert lst["total_refunded_yuan"] == 0.0
    assert _ledger("fam")["credit_yuan"] == pytest.approx(0.05, abs=1e-9)
    # 被拒单号未登记：之后同号按信用内金额可正常退付
    ok = _refund("fam", "TOO-BIG", 0.05)
    assert ok.status_code == 200, ok.text
    assert ok.json()["credit_yuan"] == 0.0


def test_refund_without_credit_or_ledger_rejected():
    # 账户只有尚欠、没有信用：任何正金额退付都整笔拒绝
    _raw_entry("owed", 1, "s1", 1, 0.10, "full")
    r = _refund("owed", "NO-CREDIT", 0.01)
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "refund_exceeds_credit"
    assert body["outstanding_yuan"] == pytest.approx(0.10, abs=1e-9)
    assert body["credit_yuan"] == 0.0
    assert _refunds("owed")["refund_count"] == 0

    # 从未入账的空账户：信用 0，同样整笔拒绝、不留记录
    r2 = _refund("never-used", "NO-LEDGER", 0.01)
    assert r2.status_code == 422
    assert r2.json()["error"] == "refund_exceeds_credit"
    assert r2.json()["credit_yuan"] == 0.0
    assert _refunds("never-used")["refund_count"] == 0


# ---------------------------------------------------------------------------
# 后续入账的应补/应退仍按原规则改变最新尚欠与信用
# ---------------------------------------------------------------------------


def test_later_supplement_after_payout_becomes_outstanding():
    """已付退款先从应退池划走：退付清空信用后，新入账的应补不再被旧应退抵减，
    直接形成尚欠；再入账的应退重新形成信用，可再次退付。"""
    sid = _make_e2e_credit("fam")
    assert _refund("fam", "RF-ALL", 0.02).status_code == 200  # 退清信用
    assert _ledger("fam")["credit_yuan"] == 0.0

    # 新更正把 t1 调高（总费用 0.08 → 0.10），凭证 4 应补 0.02
    up = dict(RECORDS[1])
    up.update(base_load_kw=0.22, grid_purchase_energy_kwh=0.02)
    assert _rev(sid, 2, [_ch(1, up)]).status_code == 200
    v4 = _issue(sid, 3).json()
    assert v4["billing"]["type"] == "supplement"
    assert v4["billing"]["amount_yuan"] == pytest.approx(0.02, abs=1e-9)
    assert _post_vouchers("fam", [(sid, 4)]).status_code == 200

    led = _ledger("fam")
    # 应退池 0.02 已被退付划走，应补 0.02 无抵减 → 尚欠 0.02、信用 0
    assert led["outstanding_yuan"] == pytest.approx(0.02, abs=1e-9)
    assert led["credit_yuan"] == 0.0
    e4 = led["entries"][3]
    assert e4["billing"]["type"] == "supplement"
    assert e4["refund_applied_yuan"] == 0.0
    assert e4["outstanding_yuan"] == pytest.approx(0.02, abs=1e-9)

    # 尚欠可正常收款（收款流程不受退付影响）
    pay = _pay("fam", "PAY-SUP", 0.02)
    assert pay.status_code == 200, pay.text
    assert [(a["entry_no"], a["allocated_yuan"]) for a in pay.json()["allocations"]] == [
        (4, pytest.approx(0.02, abs=1e-9))
    ]
    assert _ledger("fam")["outstanding_yuan"] == 0.0

    # 再更正把 t1 调回（总费用 0.10 → 0.08），凭证 5 应退 0.02 → 重新形成信用
    down = dict(RECORDS[1])
    down.update(base_load_kw=0.18, grid_purchase_energy_kwh=0.0)
    assert _rev(sid, 3, [_ch(1, down)]).status_code == 200
    v5 = _issue(sid, 4).json()
    assert v5["billing"]["type"] == "refund"
    assert _post_vouchers("fam", [(sid, 5)]).status_code == 200
    led = _ledger("fam")
    assert led["credit_yuan"] == pytest.approx(0.02, abs=1e-9)
    assert led["total_refunded_yuan"] == pytest.approx(0.02, abs=1e-9)  # 历史退付不变

    # 新信用可再次退付
    r = _refund("fam", "RF-AGAIN", 0.02)
    assert r.status_code == 200, r.text
    assert r.json()["credit_yuan"] == 0.0
    assert r.json()["total_refunded_yuan"] == pytest.approx(0.04, abs=1e-9)
    assert r.json()["refund_count"] == 2


def test_payout_does_not_rewrite_payments_or_frozen_entries():
    """退付不改写已记凭证、收款分配与首次回执：旧收款回执的逐凭证分配与
    当时快照保持冻结，台账明细的入账冻结字段不变。"""
    sid = _make_e2e_credit("fam")
    before_payments = _payments("fam")
    before_ledger = _ledger("fam")

    assert _refund("fam", "RF-1", 0.02).status_code == 200

    after_payments = _payments("fam")
    # 旧收款记录（PAY-FULL 的冻结回执）逐字段不变
    assert after_payments["payments"] == before_payments["payments"]
    # 台账明细的冻结字段（金额/计费栏/余额快照/入账时间）不变
    for before, after in zip(before_ledger["entries"], _ledger("fam")["entries"]):
        for key in ("entry_no", "session_id", "voucher_no", "voucher_id",
                    "revision_no", "billing", "amount_yuan", "total_cost_yuan",
                    "balance_after_yuan", "posted_at", "paid_yuan"):
            assert after[key] == before[key]
    # 只有账户级头寸变化：信用被扣除、累计已付退款增加
    assert after_payments["credit_yuan"] == 0.0
    assert after_payments["total_refunded_yuan"] == pytest.approx(0.02, abs=1e-9)


# ---------------------------------------------------------------------------
# 幂等重试与单号冲突
# ---------------------------------------------------------------------------


def test_same_refund_no_retry_returns_first_frozen_receipt():
    sid = _make_e2e_credit("fam")
    first = _refund("fam", "DUP", 0.01)
    assert first.status_code == 200
    first_body = first.json()
    assert first_body["result"] == "paid"

    # 之后又入账新凭证（改变当前头寸：应退池剩 0.01 抵应补 0.02 → 尚欠 0.01），
    # 重试仍返回首次回执
    up = dict(RECORDS[1])
    up.update(base_load_kw=0.22, grid_purchase_energy_kwh=0.02)
    _rev(sid, 2, [_ch(1, up)])
    _issue(sid, 3)  # supplement 0.02
    _post_vouchers("fam", [(sid, 4)])
    assert _ledger("fam")["outstanding_yuan"] == pytest.approx(0.01, abs=1e-9)

    retry = _refund("fam", "DUP", 0.01)
    assert retry.status_code == 200
    rb = retry.json()
    assert rb["result"] == "deduped"
    # 冻结字段全部等于首付（refund_count 也是当时的 1）
    for key in ("account_id", "refund_no", "amount_yuan", "paid_at",
                "outstanding_yuan", "credit_yuan", "total_refunded_yuan",
                "refund_count"):
        assert rb[key] == first_body[key]
    # 不新增退付
    assert _refunds("fam")["refund_count"] == 1


def test_same_no_different_amount_conflicts():
    _raw_entry("fam", 1, "s1", 1, 0.10, "full")
    _raw_entry("fam", 2, "s1", 2, -0.15, "refund")  # 信用 0.05
    assert _refund("fam", "XNO", 0.05).status_code == 200

    r = _refund("fam", "XNO", 0.04)
    assert r.status_code == 409
    body = r.json()
    assert body["error"] == "refund_no_conflict"
    assert body["refund_no"] == "XNO"
    assert body["account_id"] == "fam"
    assert body["amount_yuan"] == pytest.approx(0.05, abs=1e-9)
    # 冲突不写入：仍只有一笔
    assert _refunds("fam")["refund_count"] == 1


def test_same_no_cross_account_conflicts():
    _raw_entry("left", 1, "s1", 1, 0.10, "full")
    _raw_entry("left", 2, "s1", 2, -0.15, "refund")
    _raw_entry("right", 1, "s2", 1, 0.10, "full")
    _raw_entry("right", 2, "s2", 2, -0.15, "refund")
    assert _refund("left", "SHARED", 0.05).status_code == 200

    r = _refund("right", "SHARED", 0.05)
    assert r.status_code == 409
    body = r.json()
    assert body["error"] == "refund_no_conflict"
    assert body["account_id"] == "left"
    assert body["amount_yuan"] == pytest.approx(0.05, abs=1e-9)
    # 右账户没有这笔退付
    assert _refunds("right")["refund_count"] == 0


# ---------------------------------------------------------------------------
# 非法输入
# ---------------------------------------------------------------------------


def test_non_positive_amount_rejected_by_validation():
    for bad in (0.0, -1.0):
        r = client.post(
            "/accounts/fam/refunds", json={"refund_no": "Z", "amount_yuan": bad}
        )
        assert r.status_code == 422
        assert r.json()["error"] == "invalid_input"


def test_empty_or_long_refund_no_rejected():
    r = client.post("/accounts/fam/refunds", json={"refund_no": "", "amount_yuan": 1.0})
    assert r.status_code == 422
    r = client.post(
        "/accounts/fam/refunds", json={"refund_no": "x" * 129, "amount_yuan": 1.0}
    )
    assert r.status_code == 422


def test_sub_micro_amount_rounded_to_zero_rejected_without_trace():
    """亚微级金额（1e-12，6 位舍入后为 0）不得生成 0 元退付：整笔拒绝不留记录。"""
    _raw_entry("micro", 1, "s1", 1, 0.10, "full")
    _raw_entry("micro", 2, "s1", 2, -0.15, "refund")
    r = _refund("micro", "TINY", 1e-12)
    assert r.status_code == 422
    assert r.json()["error"] == "refund_amount_not_positive"
    assert _refunds("micro")["refund_count"] == 0

    # 6 位小数边界：0.025 × 2 恰好退清 0.05（舍入不产生尾差，末笔正常 200）
    assert _refund("micro", "T1", 0.025).status_code == 200
    assert _refund("micro", "T2", 0.025).status_code == 200
    assert _ledger("micro")["credit_yuan"] == 0.0


# ---------------------------------------------------------------------------
# 退付记录查询
# ---------------------------------------------------------------------------


def test_refund_list_order_and_frozen_content_vs_latest_totals():
    _raw_entry("fam", 1, "s1", 1, 0.10, "full")
    _raw_entry("fam", 2, "s1", 2, -0.15, "refund")  # 信用 0.05
    _refund("fam", "R1", 0.02)
    _refund("fam", "R2", 0.03)

    lst = _refunds("fam")
    assert lst["account_id"] == "fam"
    assert lst["refund_count"] == 2
    assert [p["refund_no"] for p in lst["refunds"]] == ["R1", "R2"]
    assert all(p["result"] == "paid" for p in lst["refunds"])
    assert lst["refunds"][0]["refund_count"] == 1
    assert lst["refunds"][1]["refund_count"] == 2
    # 列表账户级字段为当前最新值
    assert lst["total_refunded_yuan"] == pytest.approx(0.05, abs=1e-9)
    assert lst["outstanding_yuan"] == 0.0
    assert lst["credit_yuan"] == 0.0
    # 每条的快照字段为当时值（累计已付退款递增、信用递减）
    assert lst["refunds"][0]["total_refunded_yuan"] == pytest.approx(0.02, abs=1e-9)
    assert lst["refunds"][0]["credit_yuan"] == pytest.approx(0.03, abs=1e-9)
    assert lst["refunds"][1]["total_refunded_yuan"] == pytest.approx(0.05, abs=1e-9)
    assert lst["refunds"][1]["credit_yuan"] == 0.0

    # 后续入账新应退：历史退付记录冻结不变，账户级信用最新
    _raw_entry("fam", 3, "s1", 3, -0.01, "refund")
    lst2 = _refunds("fam")
    assert lst2["credit_yuan"] == pytest.approx(0.01, abs=1e-9)
    assert lst2["refunds"] == lst["refunds"]

    # 空账户查询
    empty = _refunds("never-used")
    assert empty == {
        "account_id": "never-used",
        "refund_count": 0,
        "total_refunded_yuan": 0.0,
        "outstanding_yuan": 0.0,
        "credit_yuan": 0.0,
        "refunds": [],
    }


# ---------------------------------------------------------------------------
# 并发：同单号只落一笔；不同单号串行核对信用、不重复支付
# ---------------------------------------------------------------------------


def test_concurrent_same_refund_no_only_one_lands():
    _raw_entry("race", 1, "s1", 1, 0.10, "full")
    _raw_entry("race", 2, "s1", 2, -0.15, "refund")  # 信用 0.05
    outcomes = []

    def worker():
        try:
            res = submit_refund("race", "RACE-NO", 0.05)
            outcomes.append(res["result"])
        except Exception as exc:  # noqa: BLE001
            outcomes.append(getattr(exc, "code", "error"))

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    paid = [o for o in outcomes if o == "paid"]
    deduped = [o for o in outcomes if o == "deduped"]
    assert len(paid) == 1
    assert len(deduped) == 11
    assert set(outcomes) <= {"paid", "deduped"}
    lst = _refunds("race")
    assert lst["refund_count"] == 1
    assert lst["total_refunded_yuan"] == pytest.approx(0.05, abs=1e-9)
    assert _ledger("race")["credit_yuan"] == 0.0


def test_concurrent_distinct_refunds_never_exceed_credit():
    """多笔不同单号退付并发打到同一账户：串行核对信用，累计退付恰好等于
    信用、绝不超付；其余整笔拒绝且不留记录。"""
    _raw_entry("fam", 1, "s1", 1, 0.10, "full")
    _raw_entry("fam", 2, "s1", 2, -0.20, "refund")  # 抵 0.10 后信用 0.10
    results = []

    def worker(idx):
        try:
            res = submit_refund("fam", f"C{idx}", 0.025)
            results.append(res["result"])
        except Exception as exc:  # noqa: BLE001
            results.append(getattr(exc, "code", "error"))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 信用 0.10 ÷ 0.025 = 恰好 4 笔成功，其余 4 笔信用不足整笔拒绝
    assert sorted(results) == sorted(
        ["paid"] * 4 + ["refund_exceeds_credit"] * 4
    )
    lst = _refunds("fam")
    assert lst["refund_count"] == 4
    assert lst["total_refunded_yuan"] == pytest.approx(0.10, abs=1e-9)
    assert lst["credit_yuan"] == 0.0
    # 退付序号 1..4 连续
    assert sorted(p["refund_count"] for p in lst["refunds"]) == [1, 2, 3, 4]
    assert _ledger("fam")["total_refunded_yuan"] == pytest.approx(0.10, abs=1e-9)


# ---------------------------------------------------------------------------
# 重启持久化、旧库补建、既有接口不变
# ---------------------------------------------------------------------------


def test_refunds_persist_across_restart_and_retry_idempotent(tmp_path):
    db_path = str(tmp_path / "persist-refund.db")
    configure_store(db_path)
    _raw_entry("fam", 1, "s1", 1, 0.10, "full")
    _raw_entry("fam", 2, "s1", 2, -0.15, "refund")  # 信用 0.05
    _refund("fam", "R1", 0.02)

    configure_store(db_path)  # 模拟进程重启
    lst = _refunds("fam")
    assert lst["refund_count"] == 1
    assert lst["refunds"][0]["refund_no"] == "R1"
    assert lst["total_refunded_yuan"] == pytest.approx(0.02, abs=1e-9)
    assert lst["credit_yuan"] == pytest.approx(0.03, abs=1e-9)
    led = _ledger("fam")
    assert led["credit_yuan"] == pytest.approx(0.03, abs=1e-9)
    assert led["total_refunded_yuan"] == pytest.approx(0.02, abs=1e-9)

    # 重启后同单号重试仍 deduped 且回执冻结
    retry = _refund("fam", "R1", 0.02)
    assert retry.status_code == 200
    assert retry.json()["result"] == "deduped"
    assert retry.json()["total_refunded_yuan"] == pytest.approx(0.02, abs=1e-9)
    assert _refunds("fam")["refund_count"] == 1

    # 重启后可继续退付，序号接续
    cont = _refund("fam", "R2", 0.03)
    assert cont.status_code == 200
    assert cont.json()["refund_count"] == 2
    assert cont.json()["credit_yuan"] == 0.0

    # 重启后信用不足仍整笔拒绝
    over = _refund("fam", "R3", 0.01)
    assert over.status_code == 422
    assert over.json()["error"] == "refund_exceeds_credit"


def test_legacy_db_refund_table_auto_created(tmp_path):
    """旧库（只有 sessions / session_records）打开即补建退付表：可查可退付。"""
    import json
    import sqlite3

    db_path = str(tmp_path / "legacy-refund.db")
    plan = client.post("/dispatch", json=DISPATCH).json()
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY, dispatch_json TEXT NOT NULL,
            plan_snapshot_json TEXT NOT NULL, audit_tolerance_kwh REAL NOT NULL,
            num_periods INTEGER NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE session_records (
            session_id TEXT NOT NULL, idx INTEGER NOT NULL, record_json TEXT NOT NULL,
            stored_at TEXT NOT NULL, PRIMARY KEY (session_id, idx));
        """
    )
    sid = "legacy-refund-session"
    conn.execute(
        "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?)",
        (sid, json.dumps(DISPATCH), json.dumps(plan), TOL, 4, "2026-01-01T00:00:00Z"),
    )
    for idx, rec in enumerate(RECORDS):
        conn.execute(
            "INSERT INTO session_records VALUES (?, ?, ?, ?)",
            (sid, idx, json.dumps(rec), "2026-01-01T00:00:01Z"),
        )
    conn.commit()
    conn.close()

    configure_store(db_path)
    # 退付表自动补建：空账户退付查询为 0
    assert _refunds("new-acct")["refund_count"] == 0
    # 端到端：出单 → 入账 → 收清 → 更正出应退 → 入账 → 退付
    assert _issue(sid, 0).status_code == 200
    assert _post_vouchers("new-acct", [(sid, 1)]).status_code == 200
    assert _pay("new-acct", "PAY-ALL", 0.10).status_code == 200
    down = dict(RECORDS[1])
    down.update(base_load_kw=0.18, grid_purchase_energy_kwh=0.0)
    assert _rev(sid, 0, [_ch(1, down)]).status_code == 200
    assert _issue(sid, 1).json()["billing"]["type"] == "refund"
    assert _post_vouchers("new-acct", [(sid, 2)]).status_code == 200
    assert _ledger("new-acct")["credit_yuan"] == pytest.approx(0.01, abs=1e-9)
    pay = _refund("new-acct", "LEGACY-RF", 0.01)
    assert pay.status_code == 200, pay.text
    assert pay.json()["credit_yuan"] == 0.0
    # 存储层表结构就位
    store = get_store()
    with store._connect() as c:
        names = {
            r[0]
            for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert "account_refund_payouts" in names


def test_payment_and_ledger_flows_unchanged_by_refunds():
    """新增退付不影响原收款、入账、出单行为（端到端抽验）。"""
    sid = _make_e2e_credit("fam")
    # 收款接口行为不变：无尚欠时收款仍整笔拒绝
    r = _pay("fam", "P-X", 0.01)
    assert r.status_code == 422
    assert r.json()["error"] == "payment_exceeds_outstanding"
    # 重复入账仍 409
    dup = _post_vouchers("fam", [(sid, 1)])
    assert dup.status_code == 409
    # 凭证列表仍可查
    vouchers = client.get(f"/dispatch/sessions/{sid}/vouchers")
    assert vouchers.status_code == 200
    assert [v["voucher_no"] for v in vouchers.json()["vouchers"]] == [1, 2, 3]
    # 退付后收款重试仍返回首收冻结回执
    assert _refund("fam", "RF-1", 0.02).status_code == 200
    retry = _pay("fam", "PAY-FULL", 0.10)
    assert retry.status_code == 200
    assert retry.json()["result"] == "deduped"
    assert retry.json()["total_received_yuan"] == pytest.approx(0.10, abs=1e-9)
