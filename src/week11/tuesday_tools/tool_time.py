"""
TOOL NODE 1: get_current_time
-------------------------------
WHY THIS EXISTS IN PRODUCTION:
GPT-4o has no idea what time or date it is right now.
Without this tool it cannot answer:
  - "Is the support line open right now?"
  - "Am I still in my 14-day cooling off period?"
  - "How many days until my contract ends?"

This tool gives the AI real-time awareness.
This is the same pattern used in every production AI assistant
that needs to be time-aware — banking apps, booking systems,
customer service bots.
"""

def get_current_time() -> dict:
    from datetime import datetime
    import pytz

    # Get current time in multiple timezones
    # In production you would detect the customer's timezone
    now_utc = datetime.now(pytz.UTC)
    now_uk  = datetime.now(pytz.timezone("Europe/London"))

    # Business hours logic — ConnectPlus support hours
    hour_uk = now_uk.hour
    day_uk  = now_uk.weekday()  # 0=Monday, 6=Sunday

    is_weekday      = day_uk < 5
    is_business_hours = 8 <= hour_uk < 20

    if is_weekday and is_business_hours:
        support_status = "OPEN — agents available now"
    elif is_weekday:
        support_status = "CLOSED — opens 8am UK time"
    else:
        support_status = "CLOSED — reopens Monday 8am"

    return {
        "utc_time":        now_utc.strftime("%Y-%m-%d %H:%M UTC"),
        "uk_time":         now_uk.strftime("%Y-%m-%d %H:%M GMT"),
        "uk_date":         now_uk.strftime("%A %d %B %Y"),
        "hour_uk":         hour_uk,
        "is_business_hours": is_business_hours,
        "support_status":  support_status,
        "day_name":        now_uk.strftime("%A")
    }

# Test it directly
if __name__ == "__main__":
    result = get_current_time()
    for key, value in result.items():
        print(f"  {key}: {value}")
