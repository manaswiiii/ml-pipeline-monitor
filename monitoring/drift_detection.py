import psycopg2
import numpy as np
from datetime import datetime, timedelta

def get_recent_predictions(hours=1):
    conn = psycopg2.connect("dbname=mlpipeline")
    cur = conn.cursor()
    cur.execute("""
        SELECT input_text, confidence, model_version 
        FROM predictions 
        WHERE timestamp > %s
    """, (datetime.utcnow() - timedelta(hours=hours),))
    rows = cur.fetchall()
    conn.close()
    return rows

def check_drift():
    rows = get_recent_predictions(hours=24)
    
    if len(rows) < 5:
        print("Not enough data for drift detection (need 5+ predictions)")
        return

    texts = [r[0] for r in rows]
    confidences = [r[1] for r in rows]
    
    # Check 1: text length drift
    lengths = [len(t.split()) for t in texts]
    avg_length = np.mean(lengths)
    
    # Check 2: confidence drift
    avg_confidence = np.mean(confidences)
    low_confidence = sum(1 for c in confidences if c < 0.7)
    low_conf_pct = low_confidence / len(confidences)
    
    print(f"=== Drift Detection Report ===")
    print(f"Predictions analyzed: {len(rows)}")
    print(f"Avg input length: {avg_length:.1f} words")
    print(f"Avg confidence: {avg_confidence:.3f}")
    print(f"Low confidence predictions (<0.7): {low_conf_pct:.1%}")
    
    # Alerts
    if avg_length > 50:
        print("⚠️  ALERT: Input text unusually long")
    if avg_confidence < 0.7:
        print("⚠️  ALERT: Average confidence is low — possible drift!")
    if low_conf_pct > 0.3:
        print("⚠️  ALERT: >30% low confidence predictions — model may be struggling!")
    
    if avg_confidence >= 0.7 and low_conf_pct <= 0.3:
        print("✅ No drift detected")

if __name__ == "__main__":
    check_drift()