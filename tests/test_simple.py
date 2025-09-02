"""
Simple test to verify pytest setup
간단한 pytest 설정 검증 테스트
"""

def test_simple_assertion():
    """기본 assertion 테스트"""
    assert 1 + 1 == 2
    assert "hello" == "hello"
    assert True is True

def test_simple_math():
    """간단한 수학 연산 테스트"""
    result = 5 * 3
    assert result == 15
    
    result = 10 / 2
    assert result == 5.0

def test_string_operations():
    """문자열 연산 테스트"""
    text = "Hello, World!"
    assert len(text) == 13
    assert text.startswith("Hello")
    assert text.endswith("!")

class TestBasicOperations:
    """기본 연산 테스트 클래스"""
    
    def test_list_operations(self):
        """리스트 연산 테스트"""
        numbers = [1, 2, 3, 4, 5]
        assert len(numbers) == 5
        assert numbers[0] == 1
        assert numbers[-1] == 5
        
        numbers.append(6)
        assert len(numbers) == 6
    
    def test_dict_operations(self):
        """딕셔너리 연산 테스트"""
        data = {"name": "test", "value": 42}
        assert data["name"] == "test"
        assert data["value"] == 42
        
        data["new_key"] = "new_value"
        assert "new_key" in data