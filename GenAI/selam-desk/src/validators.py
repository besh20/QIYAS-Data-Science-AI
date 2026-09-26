"""Local-context validators. This is the part no generic tutorial has."""
import re
from datetime import date
from ethiopian_date import EthiopianDateConverter

PHONE_RE = re.compile(r"^(?:\+251|0)9\d{8}$")
ETH_DATE_RE = re.compile(r"^(\d{3,4})[-/](\d{1,2})[-/](\d{1,2})$")


def validate_ethiopian_date(text: str) -> tuple[bool, str]:
    """Accepts 'YYYY-MM-DD' or 'YYYY/MM/DD' in the ETHIOPIAN calendar
    (month 1-13, Pagume=13 has 5 or 6 days). Returns (ok, normalized_string)
    where normalized_string is 'EC_YYYY-MM-DD | GC_YYYY-MM-DD' on success,
    so the stored record carries both calendars.
    """
    text = text.strip()
    m = ETH_DATE_RE.match(text)
    if not m:
        return False, text
    year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= month <= 13):
        return False, text
    max_day = 30 if month <= 12 else (6 if (year % 4 == 3) else 5)
    if not (1 <= day <= max_day):
        return False, text
    try:
        gregorian = EthiopianDateConverter.to_gregorian(year, month, day)
    except (ValueError, Exception):
        return False, text
    if gregorian > date.today():
        return False, text  # birth date can't be in the future
    normalized = f"EC_{year:04d}-{month:02d}-{day:02d} | GC_{gregorian.isoformat()}"
    return True, normalized


def validate_phone(phone: str) -> tuple[bool, str]:
    cleaned = phone.strip().replace(" ", "").replace("-", "")
    if PHONE_RE.match(cleaned):
        # normalize to +251 form
        if cleaned.startswith("0"):
            cleaned = "+251" + cleaned[1:]
        return True, cleaned
    return False, phone


def validate_name(name: str) -> tuple[bool, str]:
    name = name.strip()
    if len(name) < 2:
        return False, name
    # accept both Ge'ez (Ethiopic unicode block) and Latin script
    ok = bool(re.match(r"^[\u1200-\u137F\sA-Za-z.'-]+$", name))
    return ok, name


if __name__ == "__main__":
    print(validate_phone("0911223344"))
    print(validate_phone("+251911223344"))
    print(validate_phone("12345"))
    print(validate_name("አበበ ከበደ"))
    print(validate_name("Abebe Kebede"))
