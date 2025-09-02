"""
Basic functionality tests without heavy imports
기본 기능 테스트 (복잡한 import 없이)
"""
import pytest
import sys
import os
from pathlib import Path


class TestBasicEnvironment:
    """기본 환경 테스트"""
    
    def test_python_version(self):
        """Python 버전 테스트"""
        assert sys.version_info >= (3, 11), f"Python 3.11+ required, got {sys.version_info}"
    
    def test_app_directory_exists(self):
        """앱 디렉토리 존재 확인"""
        app_dir = Path(__file__).parent.parent / "app"
        assert app_dir.exists(), "app directory does not exist"
        assert app_dir.is_dir(), "app is not a directory"
    
    def test_main_file_exists(self):
        """메인 파일 존재 확인"""
        main_file = Path(__file__).parent.parent / "main.py"
        assert main_file.exists(), "main.py does not exist"
    
    def test_pyproject_file_exists(self):
        """pyproject.toml 파일 존재 확인"""
        pyproject_file = Path(__file__).parent.parent / "pyproject.toml"
        assert pyproject_file.exists(), "pyproject.toml does not exist"


class TestBasicImports:
    """기본 import 테스트"""
    
    def test_stdlib_imports(self):
        """표준 라이브러리 import 테스트"""
        import json
        import os
        import sys
        import pathlib
        import asyncio
        
        assert json is not None
        assert os is not None
        assert sys is not None
        assert pathlib is not None
        assert asyncio is not None
    
    def test_fastapi_import(self):
        """FastAPI import 테스트"""
        import fastapi
        from fastapi import FastAPI
        
        app = FastAPI()
        assert app is not None
        assert isinstance(app, FastAPI)
    
    def test_pytest_import(self):
        """pytest import 테스트"""
        import pytest
        assert pytest is not None
    
    def test_yaml_import(self):
        """YAML import 테스트"""
        import yaml
        
        test_yaml = "test: value"
        parsed = yaml.safe_load(test_yaml)
        assert parsed == {"test": "value"}


class TestBasicFunctionality:
    """기본 기능 테스트"""
    
    def test_simple_function(self):
        """간단한 함수 테스트"""
        def add(a, b):
            return a + b
        
        result = add(2, 3)
        assert result == 5
    
    @pytest.mark.asyncio
    async def test_async_function(self):
        """비동기 함수 테스트"""
        import asyncio
        
        async def async_add(a, b):
            await asyncio.sleep(0.001)  # 1ms
            return a + b
        
        result = await async_add(3, 4)
        assert result == 7
    
    def test_path_operations(self):
        """경로 작업 테스트"""
        from pathlib import Path
        
        current_dir = Path(__file__).parent
        assert current_dir.exists()
        assert current_dir.is_dir()
        
        # 상위 디렉토리
        parent = current_dir.parent
        assert parent.exists()


class TestEnvironmentVariables:
    """환경 변수 테스트"""
    
    def test_environment_setup(self):
        """환경 변수 설정 테스트"""
        # conftest.py에서 설정한 환경 변수들 확인
        assert os.getenv("ENVIRONMENT") == "test"
        assert os.getenv("DEBUG") == "true"
        assert os.getenv("GOOGLE_API_KEY") is not None
    
    def test_env_variable_conversion(self):
        """환경 변수 타입 변환 테스트"""
        def convert_env_value(value: str):
            if value.lower() in ('true', 'false'):
                return value.lower() == 'true'
            try:
                if '.' in value:
                    return float(value)
                else:
                    return int(value)
            except ValueError:
                return value
        
        assert convert_env_value("true") is True
        assert convert_env_value("false") is False
        assert convert_env_value("123") == 123
        assert convert_env_value("12.34") == 12.34
        assert convert_env_value("test") == "test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])