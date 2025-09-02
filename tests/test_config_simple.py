"""
Simple Config Tests
간단한 설정 테스트
"""
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open
import yaml


class TestConfigLoader:
    """간단한 설정 로더 테스트"""
    
    def test_config_loader_import(self):
        """설정 로더 import 테스트"""
        from app.lib.config_loader import ConfigLoader
        assert ConfigLoader is not None
    
    def test_config_loader_creation(self):
        """설정 로더 생성 테스트"""
        from app.lib.config_loader import ConfigLoader
        
        loader = ConfigLoader()
        assert loader is not None
        assert hasattr(loader, 'base_path')
        assert hasattr(loader, 'environment')
    
    def test_config_path_setup(self):
        """설정 경로 설정 테스트"""
        from app.lib.config_loader import ConfigLoader
        
        loader = ConfigLoader()
        # 경로가 올바르게 설정되는지 확인
        assert loader.base_path.name == "config"
        assert str(loader.base_path).endswith("app/config")
    
    def test_yaml_file_loading(self):
        """YAML 파일 로딩 메서드 테스트"""
        from app.lib.config_loader import ConfigLoader
        
        loader = ConfigLoader()
        
        # Mock YAML 파일 내용
        mock_yaml_content = """
app:
  name: "Test App"
  version: "1.0.0"
server:
  port: 8000
"""
        
        with patch("builtins.open", mock_open(read_data=mock_yaml_content)):
            with patch.object(Path, 'exists', return_value=True):
                result = loader._load_yaml_file(Path("test.yaml"))
                
                assert result is not None
                assert "app" in result
                assert result["app"]["name"] == "Test App"
                assert result["server"]["port"] == 8000
    
    def test_config_merging(self):
        """설정 병합 테스트"""
        from app.lib.config_loader import ConfigLoader
        
        loader = ConfigLoader()
        
        base_config = {
            "app": {"name": "Base App", "version": "1.0.0"},
            "server": {"port": 8000, "host": "localhost"}
        }
        
        override_config = {
            "app": {"name": "Override App"},
            "server": {"port": 9000}
        }
        
        merged = loader._merge_configs(base_config, override_config)
        
        assert merged["app"]["name"] == "Override App"  # 오버라이드됨
        assert merged["app"]["version"] == "1.0.0"      # 유지됨
        assert merged["server"]["port"] == 9000         # 오버라이드됨
        assert merged["server"]["host"] == "localhost"  # 유지됨
    
    def test_value_conversion(self):
        """값 타입 변환 테스트"""
        from app.lib.config_loader import ConfigLoader
        
        loader = ConfigLoader()
        
        # Boolean 변환
        assert loader._convert_value("true") is True
        assert loader._convert_value("false") is False
        assert loader._convert_value("True") is True
        assert loader._convert_value("FALSE") is False
        
        # 숫자 변환
        assert loader._convert_value("123") == 123
        assert loader._convert_value("12.34") == 12.34
        
        # 문자열 유지
        assert loader._convert_value("hello") == "hello"


class TestConfigFile:
    """설정 파일 테스트"""
    
    def test_config_file_exists(self):
        """설정 파일 존재 확인"""
        config_path = Path(__file__).parent.parent / "app" / "config" / "config.yaml"
        assert config_path.exists(), f"Config file not found: {config_path}"
    
    def test_config_file_valid_yaml(self):
        """설정 파일이 유효한 YAML인지 확인"""
        config_path = Path(__file__).parent.parent / "app" / "config" / "config.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        assert config is not None
        assert isinstance(config, dict)
        
        # 기본 설정 구조 확인
        expected_keys = ["app", "server", "qdrant", "embeddings", "llm"]
        for key in expected_keys:
            assert key in config, f"Missing config key: {key}"
    
    def test_config_structure(self):
        """설정 구조 검증"""
        config_path = Path(__file__).parent.parent / "app" / "config" / "config.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # App 설정
        assert "name" in config["app"]
        assert "version" in config["app"]
        
        # Server 설정
        assert "host" in config["server"]
        assert "port" in config["server"]
        
        # Qdrant 설정
        assert "url" in config["qdrant"]
        assert "collection_name" in config["qdrant"]
        
        # Embeddings 설정
        assert "provider" in config["embeddings"]
        assert "model" in config["embeddings"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])