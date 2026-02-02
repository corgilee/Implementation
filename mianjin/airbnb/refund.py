'''
Rule 1: Any refund should be issued in full for a payment before considering the next payment.
Rule 2: Refund will be prioritized among different payment methods based on such order: CREDIT, CREDIT_CARD, PAYPAL.
Rule 3: Refunds should be issued to more recent payments.
Example 1
Input: Transactions: [(Payment1: Credit, 2023-01-10, $40), (Payment2: Paypal, 2023-01-15, $60) ],
Refund amount: $50
Output:
Refund a: linked to Payment1, Credit, $40
Refund b: linked to Payment2, Paypal, $10
Example 2
Input: Transactions: [(Payment1: Credit, 2023-01-15, $40), (Payment2: Paypal, 2023-01-10, $60),(Payment3: Paypal, 2023-01-20, $40),(Refund1: linked to Payment1, $20) ],
Refund amount: $50
Output:
Refund a: linked to Payment1, credit, $20
Refund b: linked to Payment3, Paypal, $30

heap题
'''

import heapq
from datetime import datetime

METHOD_RANK = {
    "CREDIT": 0,
    "CREDIT_CARD": 1,
    "PAYPAL": 2,
}

def _ts_to_sortable(ts):
    """
    这段code 是处理时间戳的
    Returns an integer sortable key where larger means more recent.
    Supports:
      - 'YYYY-MM-DD'
      - ISO-ish strings (best-effort)
      - int/float already
    """
    if ts is None:
        return 0
    if isinstance(ts, (int, float)):
        return int(ts)
    s = str(ts)
    # Fast path: YYYY-MM-DD
    try:
        dt = datetime.strptime(s[:10], "%Y-%m-%d")
        return int(dt.timestamp())
    except Exception:
        pass

    return 0

def allocate_refunds(transactions, refund_amount):
    # 1) Collect payments
    payments = {}  # payment_id -> dict with method, ts_key, original_amount
    remaining = {} # payment_id -> refundable remaining

    for t in transactions:
        if t.get("kind") == "PAYMENT":
            pid = t["id"]
            method = t["method"]
            if method not in METHOD_RANK:
                raise ValueError(f"Unknown method: {method}")
            ts_key = _ts_to_sortable(t["timestamp"])
            amt = t["amount"]
            payments[pid] = {"method": method, "ts_key": ts_key, "amount": amt}
            remaining[pid] = amt

    # 2) Apply existing refunds
    for t in transactions:
        if t.get("kind") == "REFUND":
            pid = t["linked_payment_id"]
            amt = t["amount"]
            if pid in remaining:
                remaining[pid] -= amt
            # else: ignore or raise depending on spec

    # Clamp negatives (over-refunded historical data)
    for pid in list(remaining.keys()):
        if remaining[pid] < 0:
            remaining[pid] = 0

    # 3) Build heap: (method_rank, -ts_key, tie, payment_id)
    heap = []
    tie = 0
    for pid, p in payments.items():
        if remaining.get(pid, 0) > 0:
            #-p["ts_key"], 是 时间越新（越大）越早弹出
            heapq.heappush(heap, (METHOD_RANK[p["method"]], -p["ts_key"], tie, pid))
            tie += 1

    # 4) Allocate
    out = []
    to_refund = refund_amount

    while to_refund > 0 and heap:
        _, _, _, pid = heapq.heappop(heap)
        avail = remaining[pid]
        if avail <= 0:
            continue

        if avail <= to_refund:
            take = avail
        else:
            take = to_refund
            
        remaining[pid] -= take
        to_refund -= take

        out.append({
            "linked_payment_id": pid,
            "method": payments[pid]["method"],
            "amount": take,
        })

        # If still refundable, push back (same priority)
        if remaining[pid] > 0:
            p = payments[pid]
            heapq.heappush(heap, (METHOD_RANK[p["method"]], -p["ts_key"], tie, pid))
            tie += 1

    # If spec requires error on insufficient refundable:
    # if to_refund > 0: raise ValueError("Insufficient refundable balance")

    return out


##### test case #####
transactions = [
    {"kind": "PAYMENT", "id": "p1", "method": "CREDIT", "timestamp": "2023-01-15", "amount": 40},
    {"kind": "PAYMENT", "id": "p2", "method": "PAYPAL", "timestamp": "2023-01-10", "amount": 60},
    {"kind": "PAYMENT", "id": "p3", "method": "PAYPAL", "timestamp": "2023-01-20", "amount": 40},

    # 👇 HISTORICAL refund (already happened in the past)
    {"kind": "REFUND", "id": "r1", "linked_payment_id": "p1", "amount": 20}
]

refund_amount = 50

print(allocate_refunds(transactions,refund_amount))