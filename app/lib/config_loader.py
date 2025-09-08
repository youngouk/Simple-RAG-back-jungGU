"""
Configuration loader for RAG Chatbot
YAML 기반 계층적 설정 로더
"""
import os
import re
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """설정 로더 클래스"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent.parent / "config"
        self.environment = os.getenv("NODE_ENV", "development")
        
    def load_config(self) -> Dict[str, Any]:
        """설정 로드 및 병합"""
        try:
            # 1. 기본 설정 로드
            base_config = self._load_yaml_file(self.base_path / "config.yaml")
            
            # 2. 환경별 설정 로드 및 병합
            env_config_path = self.base_path / "environments" / f"{self.environment}.yaml"
            if env_config_path.exists():
                env_config = self._load_yaml_file(env_config_path)
                base_config = self._merge_configs(base_config, env_config)
            
            # 3. 환경 변수 치환 적용
            base_config = self._substitute_env_vars(base_config)
            
            # 4. 환경 변수 오버라이드 적용
            base_config = self._apply_env_overrides(base_config)
            
            return base_config
            
        except Exception as e:
            raise Exception(f"Failed to load configuration: {e}")
    
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """YAML 파일 로드"""
        if not file_path.exists():
            return {}
            
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """설정 깊은 병합"""
        merged = base.copy()
        
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
                
        return merged
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """환경 변수 오버라이드 적용"""
        # 주요 환경 변수들
        env_mappings = {
            'PORT': ('server', 'port'),
            'HOST': ('server', 'host'),
            'GOOGLE_API_KEY': ('llm', 'google', 'api_key'),
            'OPENAI_API_KEY': ('llm', 'openai', 'api_key'),
            'ANTHROPIC_API_KEY': ('llm', 'anthropic', 'api_key'),
            'COHERE_API_KEY': ('reranking', 'providers', 'cohere', 'api_key'),
            'QDRANT_URL': ('qdrant', 'url'),
            'QDRANT_API_KEY': ('qdrant', 'api_key'),
            'QDRANT_HOST': ('qdrant', 'host'),
            'QDRANT_PORT': ('qdrant', 'port'),
            'REDIS_URL': ('session', 'redis_url'),
            'LOG_LEVEL': ('logging', 'level'),
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested_value(config, config_path, value)
                
        return config
    
    def _set_nested_value(self, config: Dict[str, Any], path: tuple, value: str):
        """중첩된 딕셔너리에 값 설정"""
        current = config
        
        # 경로의 마지막 키까지 탐색하며 딕셔너리 생성
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
            
        # 타입 변환
        converted_value = self._convert_value(value)
        current[path[-1]] = converted_value
    
    def _convert_value(self, value: str) -> Any:
        """환경 변수 값 타입 변환"""
        # Boolean 변환
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
            
        # 숫자 변환
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
            
        return value
    
    def _substitute_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """환경 변수 치환 적용"""
        def substitute_value(value):
            if isinstance(value, str):
                # ${VAR} 또는 ${VAR:-default} 형식 치환
                pattern = r'\$\{([^}]+)\}'
                
                def replace_env_var(match):
                    var_expr = match.group(1)
                    if ':-' in var_expr:
                        var_name, default_value = var_expr.split(':-', 1)
                        return os.getenv(var_name, default_value)
                    else:
                        return os.getenv(var_expr, match.group(0))
                
                return re.sub(pattern, replace_env_var, value)
            elif isinstance(value, dict):
                return {k: substitute_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [substitute_value(item) for item in value]
            else:
                return value
        
        return substitute_value(config)


def load_config() -> Dict[str, Any]:
    """전역 설정 로드 함수"""
    loader = ConfigLoader()
    return loader.load_config()