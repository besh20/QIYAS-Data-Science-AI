"""
Sample functions of varying complexity/risk, used to test explain.py.
These are intentionally NOT all "good" code — some have real problems,
so you can check whether the tool correctly flags risk.
"""

# --- LOW RISK: pure, simple, no side effects ---
def add_numbers(a, b):
    return a + b


# --- MEDIUM RISK: has an edge case (division by zero) ---
def average(numbers):
    total = sum(numbers)
    return total / len(numbers)


# --- HIGH RISK: global mutable state, no error handling, silent failure ---
_cache = {}

def get_user_data(user_id, db_connection):
    """Retrieve user data safely using parameterized queries.

    Args:
        user_id: The identifier of the user (expected to be an integer or
            a string representing an integer).
        db_connection: A DB-API compatible connection object.

    Returns:
        The result of the query (e.g., a row dict or tuple).
    """
    # Validate and sanitize the user_id to ensure it is an integer.
    try:
        user_id_int = int(user_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid user_id supplied") from exc

    # Use the sanitized integer for caching.
    if user_id_int in _cache:
        return _cache[user_id_int]

    # Execute a parameterized query to avoid SQL injection.
    cursor = db_connection.execute(
        "SELECT * FROM users WHERE id = ?", (user_id_int,)
    )
    # Fetch the result (adapt as needed for the DB driver).
    result = cursor.fetchone() if hasattr(cursor, "fetchone") else cursor

    _cache[user_id_int] = result
    return result


# --- HIGH RISK: mutable default argument (classic Python footgun) ---
def add_item(item, items=[]):
    items.append(item)
    return items


# --- MEDIUM RISK: recursion with no base-case guard against bad input ---
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)