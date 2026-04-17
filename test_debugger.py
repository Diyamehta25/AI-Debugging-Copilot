# test_debugger.py
# pytest test suite for AI Debugging Copilot
# Run with: pytest test_debugger.py -v

import pytest
from debugger import debug_with_langchain


class TestValidCode:
    """Tests for code that runs without errors"""

    def test_simple_print_no_error(self):
        code = "x = 5\nprint(x)"
        result = debug_with_langchain(code)
        assert "No errors found" in result

    def test_list_operations_no_error(self):
        code = "nums = [1,2,3]\nprint(sum(nums))"
        result = debug_with_langchain(code)
        assert "No errors found" in result


class TestErrorDetection:
    """Tests for each supported error type"""

    def test_index_error_detected(self):
        code = "arr = []\nprint(arr[0])"
        result = debug_with_langchain(code)
        assert "IndexError" in result

    def test_zero_division_detected(self):
        code = "x = 10 / 0"
        result = debug_with_langchain(code)
        assert "ZeroDivisionError" in result

    def test_type_error_detected(self):
        code = "result = 'hello' + 5"
        result = debug_with_langchain(code)
        assert "TypeError" in result

    def test_name_error_detected(self):
        code = "print(undefined_var)"
        result = debug_with_langchain(code)
        assert "NameError" in result

    def test_key_error_detected(self):
        code = "d = {'a': 1}\nprint(d['z'])"
        result = debug_with_langchain(code)
        assert "KeyError" in result

    def test_attribute_error_detected(self):
        code = "x = 5\nx.append(1)"
        result = debug_with_langchain(code)
        assert "AttributeError" in result


class TestOutputStructure:
    """Tests that output always has correct structure"""

    def test_result_contains_error_label(self):
        code = "arr = []\nprint(arr[0])"
        result = debug_with_langchain(code)
        assert "Error" in result

    def test_result_contains_reason_label(self):
        code = "arr = []\nprint(arr[0])"
        result = debug_with_langchain(code)
        assert "Reason" in result

    def test_result_contains_fix_label(self):
        code = "arr = []\nprint(arr[0])"
        result = debug_with_langchain(code)
        assert "Fix" in result

    def test_result_is_string(self):
        code = "x = 10/0"
        result = debug_with_langchain(code)
        assert isinstance(result, str)


class TestEdgeCases:
    """Tests for edge cases and unusual inputs"""

    def test_empty_string_input(self):
        result = debug_with_langchain("")
        assert result is not None
        assert "Error" in result

    def test_whitespace_only_input(self):
        result = debug_with_langchain("   \n  ")
        assert result is not None

    def test_multiline_code(self):
        code = "def foo():\n    arr = []\n    return arr[5]\nfoo()"
        result = debug_with_langchain(code)
        assert "IndexError" in result

    def test_none_not_returned(self):
        code = "x = 1/0"
        result = debug_with_langchain(code)
        assert result is not None