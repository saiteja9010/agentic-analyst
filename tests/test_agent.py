import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent import unescape_sql_operators  # noqa: E402


def test_unescapes_json_unicode_lt():
    query = 'SELECT * FROM order_items WHERE profit \\u003c 0'
    assert unescape_sql_operators(query) == "SELECT * FROM order_items WHERE profit < 0"


def test_unescapes_json_unicode_gt():
    query = 'SELECT * FROM order_items WHERE profit \\u003e 0'
    assert unescape_sql_operators(query) == "SELECT * FROM order_items WHERE profit > 0"


def test_unescapes_html_entities():
    query = "SELECT * FROM order_items WHERE profit &lt; 0 &amp; amount &gt; 100"
    assert unescape_sql_operators(query) == "SELECT * FROM order_items WHERE profit < 0 & amount > 100"


def test_leaves_normal_sql_untouched():
    query = "SELECT * FROM order_items WHERE profit < 0 AND amount > 100"
    assert unescape_sql_operators(query) == query


def test_mixed_escaping_in_one_query():
    query = 'SELECT category FROM order_items WHERE profit \\u003c 0 AND amount &gt; 50'
    assert (
        unescape_sql_operators(query)
        == "SELECT category FROM order_items WHERE profit < 0 AND amount > 50"
    )
