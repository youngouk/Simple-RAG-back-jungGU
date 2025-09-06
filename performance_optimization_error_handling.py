# 성능 최적화 및 에러 핸들링 설계
# Performance Optimization and Error Handling Design

"""
Enhanced RAG 시스템의 성능 최적화 및 에러 핸들링 전략

이 모듈은 다음과 같은 최적화 및 에러 처리 메커니즘을 제공합니다:
1. 캐싱 전략 및 메모리 관리
2. 비동기 처리 및 병렬화
3. 회로 차단기 패턴 (Circuit Breaker)
4. 재시도 메커니즘 (Retry Logic)
5. 성능 모니터링 및 알림
6. 리소스 할당 및 제한
7. 그레이스풀 디그레데이션 (Graceful Degradation)
"""

import asyncio
import time
import logging
import hashlib
import weakref
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from enum import Enum
import json
import traceback

logger = logging.getLogger(__name__)

# =====================================
# 1. 캐싱 시스템 설계
# =====================================

class CachePolicy(Enum):
    """캐시 정책"""
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used  
    TTL = "ttl"           # Time To Live
    HYBRID = "hybrid"     # TTL + LRU 조합

@dataclass
class CacheEntry:
    """캐시 엔트리"""
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    ttl: Optional[int] = None
    size: int = 0
    
    def is_expired(self) -> bool:
        """만료 여부 확인"""
        if not self.ttl:
            return False
        return datetime.now() - self.created_at > timedelta(seconds=self.ttl)
    
    def touch(self):
        """액세스 시간 및 카운트 업데이트"""
        self.accessed_at = datetime.now()
        self.access_count += 1

class IntelligentCache:
    """지능형 캐시 시스템"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600, 
                 policy: CachePolicy = CachePolicy.HYBRID):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.policy = policy
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order = deque()  # LRU용
        
        # 통계
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'errors': 0,
            'total_size': 0,
            'avg_access_time': 0
        }
    
    async def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 조회"""
        try:
            entry = self.cache.get(key)
            
            if not entry:
                self.stats['misses'] += 1
                return None
            
            # 만료 확인
            if entry.is_expired():
                await self.delete(key)
                self.stats['misses'] += 1
                return None
            
            # 액세스 정보 업데이트
            entry.touch()
            
            # LRU 순서 업데이트
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            self.stats['hits'] += 1
            return entry.value
            
        except Exception as e:
            logger.error(f"캐시 조회 실패: {key} - {e}")
            self.stats['errors'] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """캐시에 값 저장"""
        try:
            # TTL 설정
            cache_ttl = ttl or self.default_ttl
            
            # 엔트리 생성
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                accessed_at=datetime.now(),
                ttl=cache_ttl,
                size=self._calculate_size(value)
            )
            
            # 크기 확인 및 eviction
            if len(self.cache) >= self.max_size:
                await self._evict()
            
            # 저장
            self.cache[key] = entry
            self.access_order.append(key)
            self.stats['total_size'] += entry.size
            
            return True
            
        except Exception as e:
            logger.error(f"캐시 저장 실패: {key} - {e}")
            self.stats['errors'] += 1
            return False
    
    async def delete(self, key: str) -> bool:
        """캐시에서 삭제"""
        try:
            if key in self.cache:
                entry = self.cache.pop(key)
                self.stats['total_size'] -= entry.size
                
                if key in self.access_order:
                    self.access_order.remove(key)
                
                return True
            return False
            
        except Exception as e:
            logger.error(f"캐시 삭제 실패: {key} - {e}")
            self.stats['errors'] += 1
            return False
    
    async def _evict(self) -> str:
        """캐시 엔트리 제거"""
        if not self.cache:
            return ""
        
        if self.policy == CachePolicy.LRU:
            # LRU: 가장 오래된 액세스
            evict_key = self.access_order.popleft()
        elif self.policy == CachePolicy.LFU:
            # LFU: 가장 적게 액세스된 것
            evict_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k].access_count)
        elif self.policy == CachePolicy.TTL:
            # TTL: 만료된 것 우선, 없으면 가장 오래된 것
            expired_keys = [k for k, v in self.cache.items() if v.is_expired()]
            if expired_keys:
                evict_key = expired_keys[0]
            else:
                evict_key = min(self.cache.keys(),
                               key=lambda k: self.cache[k].created_at)
        else:  # HYBRID
            # TTL + LRU 조합
            expired_keys = [k for k, v in self.cache.items() if v.is_expired()]
            if expired_keys:
                evict_key = expired_keys[0]
            else:
                evict_key = self.access_order.popleft()
        
        await self.delete(evict_key)
        self.stats['evictions'] += 1
        return evict_key
    
    def _calculate_size(self, value: Any) -> int:
        """객체 크기 계산 (대략적)"""
        try:
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (list, dict)):
                return len(str(value).encode('utf-8'))
            else:
                return len(str(value).encode('utf-8'))
        except:
            return 100  # 기본값
    
    def get_hit_rate(self) -> float:
        """캐시 적중률 계산"""
        total = self.stats['hits'] + self.stats['misses']
        if total == 0:
            return 0.0
        return self.stats['hits'] / total
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        return {
            **self.stats,
            'cache_size': len(self.cache),
            'max_size': self.max_size,
            'hit_rate': self.get_hit_rate(),
            'policy': self.policy.value
        }

# =====================================
# 2. 회로 차단기 패턴 (Circuit Breaker)
# =====================================

class CircuitState(Enum):
    """회로 상태"""
    CLOSED = "closed"       # 정상 동작
    OPEN = "open"           # 차단됨
    HALF_OPEN = "half_open" # 테스트 중

@dataclass
class CircuitBreakerConfig:
    """회로 차단기 설정"""
    failure_threshold: int = 5          # 실패 임계값
    success_threshold: int = 3          # 복구 임계값  
    timeout: int = 60                   # 차단 시간 (초)
    expected_exception: type = Exception # 모니터링할 예외 타입

class CircuitBreaker:
    """회로 차단기"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'state_changes': 0
        }
    
    @asynccontextmanager
    async def call(self):
        """회로 차단기를 통한 호출"""
        self.stats['total_calls'] += 1
        
        # 상태별 처리
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._set_state(CircuitState.HALF_OPEN)
            else:
                raise Exception(f"Circuit breaker {self.name} is OPEN")
        
        try:
            yield
            # 성공 처리
            self._on_success()
            
        except self.config.expected_exception as e:
            # 실패 처리
            self._on_failure()
            raise e
    
    def _on_success(self):
        """성공 시 처리"""
        self.stats['successful_calls'] += 1
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._set_state(CircuitState.CLOSED)
                self.failure_count = 0
                self.success_count = 0
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0
    
    def _on_failure(self):
        """실패 시 처리"""
        self.stats['failed_calls'] += 1
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self._set_state(CircuitState.OPEN)
    
    def _should_attempt_reset(self) -> bool:
        """재시도 가능 여부 확인"""
        if not self.last_failure_time:
            return True
        
        return time.time() - self.last_failure_time >= self.config.timeout
    
    def _set_state(self, new_state: CircuitState):
        """상태 변경"""
        if self.state != new_state:
            logger.info(f"Circuit breaker {self.name}: {self.state.value} → {new_state.value}")
            self.state = new_state
            self.stats['state_changes'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 반환"""
        return {
            **self.stats,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'config': {
                'failure_threshold': self.config.failure_threshold,
                'success_threshold': self.config.success_threshold,
                'timeout': self.config.timeout
            }
        }

# =====================================
# 3. 재시도 메커니즘
# =====================================

@dataclass
class RetryConfig:
    """재시도 설정"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True

class RetryManager:
    """재시도 관리자"""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_retries': 0
        }
    
    async def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """재시도가 있는 함수 실행"""
        self.stats['total_operations'] += 1
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                if attempt > 0:
                    self.stats['total_retries'] += attempt
                
                self.stats['successful_operations'] += 1
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"재시도 {attempt + 1}/{self.config.max_attempts}: "
                        f"{delay:.2f}초 대기 후 재시도 - {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"모든 재시도 실패: {e}")
        
        self.stats['failed_operations'] += 1
        if last_exception:
            raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """지연 시간 계산 (지수 백오프 + 지터)"""
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        delay = min(delay, self.config.max_delay)
        
        if self.config.jitter:
            # ±25% 지터 추가
            import random
            jitter = delay * 0.25 * (2 * random.random() - 1)
            delay += jitter
        
        return max(0, delay)
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 반환"""
        success_rate = 0
        if self.stats['total_operations'] > 0:
            success_rate = self.stats['successful_operations'] / self.stats['total_operations']
        
        return {
            **self.stats,
            'success_rate': success_rate,
            'avg_retries_per_operation': (
                self.stats['total_retries'] / max(1, self.stats['total_operations'])
            )
        }

# =====================================
# 4. 성능 모니터링 시스템
# =====================================

@dataclass
class MetricData:
    """메트릭 데이터"""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)

class PerformanceMonitor:
    """성능 모니터링"""
    
    def __init__(self, alert_threshold: Dict[str, float] = None):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alert_threshold = alert_threshold or {}
        self.alerts = []
        
        # 시스템 메트릭
        self.system_stats = {
            'start_time': time.time(),
            'request_count': 0,
            'error_count': 0,
            'avg_response_time': 0,
            'memory_usage': 0
        }
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        """메트릭 기록"""
        metric = MetricData(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {}
        )
        
        self.metrics[name].append(metric)
        
        # 알림 임계값 확인
        if name in self.alert_threshold:
            if value > self.alert_threshold[name]:
                self._create_alert(name, value)
    
    @asynccontextmanager
    async def measure_time(self, operation_name: str):
        """실행 시간 측정"""
        start_time = time.time()
        self.system_stats['request_count'] += 1
        
        try:
            yield
        except Exception as e:
            self.system_stats['error_count'] += 1
            self.record_metric(f"{operation_name}_error", 1)
            raise
        finally:
            duration = time.time() - start_time
            self.record_metric(f"{operation_name}_duration", duration)
            
            # 평균 응답 시간 업데이트
            self._update_avg_response_time(duration)
    
    def _update_avg_response_time(self, duration: float):
        """평균 응답 시간 업데이트"""
        count = self.system_stats['request_count']
        current_avg = self.system_stats['avg_response_time']
        
        # 이동 평균 계산
        self.system_stats['avg_response_time'] = (
            (current_avg * (count - 1) + duration) / count
        )
    
    def _create_alert(self, metric_name: str, value: float):
        """알림 생성"""
        alert = {
            'metric': metric_name,
            'value': value,
            'threshold': self.alert_threshold[metric_name],
            'timestamp': datetime.now(),
            'severity': 'warning' if value < self.alert_threshold[metric_name] * 1.5 else 'critical'
        }
        
        self.alerts.append(alert)
        logger.warning(f"성능 알림: {metric_name}={value} (임계값: {self.alert_threshold[metric_name]})")
    
    def get_metric_stats(self, name: str, window_minutes: int = 10) -> Dict[str, float]:
        """메트릭 통계 계산"""
        if name not in self.metrics:
            return {}
        
        # 시간 윈도우 필터링
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_metrics = [
            m for m in self.metrics[name] 
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        values = [m.value for m in recent_metrics]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'p50': self._percentile(values, 0.5),
            'p95': self._percentile(values, 0.95),
            'p99': self._percentile(values, 0.99)
        }
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """백분위수 계산"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * percentile
        f = int(k)
        c = k - f
        
        if f == len(sorted_values) - 1:
            return sorted_values[f]
        else:
            return sorted_values[f] * (1 - c) + sorted_values[f + 1] * c
    
    def get_system_health(self) -> Dict[str, Any]:
        """시스템 헬스 체크"""
        uptime = time.time() - self.system_stats['start_time']
        error_rate = 0
        
        if self.system_stats['request_count'] > 0:
            error_rate = self.system_stats['error_count'] / self.system_stats['request_count']
        
        health_status = "healthy"
        if error_rate > 0.05:  # 5% 이상
            health_status = "degraded"
        if error_rate > 0.1:   # 10% 이상
            health_status = "unhealthy"
        
        return {
            'status': health_status,
            'uptime': uptime,
            'error_rate': error_rate,
            'avg_response_time': self.system_stats['avg_response_time'],
            'total_requests': self.system_stats['request_count'],
            'recent_alerts': len([a for a in self.alerts if 
                                datetime.now() - a['timestamp'] < timedelta(minutes=30)])
        }

# =====================================
# 5. 리소스 관리 및 제한
# =====================================

class ResourceLimiter:
    """리소스 제한 관리자"""
    
    def __init__(self, max_concurrent: int = 100, max_memory_mb: int = 1024,
                 max_cpu_percent: float = 80.0):
        self.max_concurrent = max_concurrent
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
        
        self.current_concurrent = 0
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # 리소스 사용량 통계
        self.stats = {
            'peak_concurrent': 0,
            'total_requests': 0,
            'rejected_requests': 0,
            'avg_wait_time': 0
        }
    
    @asynccontextmanager
    async def acquire(self, operation_name: str = "default"):
        """리소스 획득"""
        wait_start = time.time()
        
        # 리소스 제한 확인
        if not await self._check_resource_availability():
            self.stats['rejected_requests'] += 1
            raise Exception("리소스 한계 초과: 요청이 거부되었습니다")
        
        try:
            await self.semaphore.acquire()
            wait_time = time.time() - wait_start
            
            self.current_concurrent += 1
            self.stats['peak_concurrent'] = max(
                self.stats['peak_concurrent'], 
                self.current_concurrent
            )
            self.stats['total_requests'] += 1
            
            # 평균 대기 시간 업데이트
            self._update_avg_wait_time(wait_time)
            
            logger.debug(f"리소스 획득: {operation_name} (동시: {self.current_concurrent})")
            yield
            
        finally:
            self.current_concurrent -= 1
            self.semaphore.release()
            logger.debug(f"리소스 해제: {operation_name}")
    
    async def _check_resource_availability(self) -> bool:
        """리소스 가용성 확인"""
        # 메모리 사용량 확인 (간단한 구현)
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            if memory_percent > 90:  # 90% 이상 사용
                logger.warning(f"메모리 사용량 높음: {memory_percent}%")
                return False
            
            if cpu_percent > self.max_cpu_percent:
                logger.warning(f"CPU 사용량 높음: {cpu_percent}%")
                return False
                
        except ImportError:
            # psutil 없으면 기본 허용
            pass
        
        return True
    
    def _update_avg_wait_time(self, wait_time: float):
        """평균 대기 시간 업데이트"""
        count = self.stats['total_requests']
        current_avg = self.stats['avg_wait_time']
        
        self.stats['avg_wait_time'] = (
            (current_avg * (count - 1) + wait_time) / count
        )
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """리소스 통계"""
        utilization = self.current_concurrent / self.max_concurrent if self.max_concurrent > 0 else 0
        rejection_rate = 0
        
        if self.stats['total_requests'] > 0:
            rejection_rate = self.stats['rejected_requests'] / (
                self.stats['total_requests'] + self.stats['rejected_requests']
            )
        
        return {
            'current_concurrent': self.current_concurrent,
            'max_concurrent': self.max_concurrent,
            'utilization': utilization,
            'rejection_rate': rejection_rate,
            **self.stats
        }

# =====================================
# 6. 통합 성능 최적화 매니저
# =====================================

class OptimizationManager:
    """통합 성능 최적화 매니저"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 컴포넌트 초기화
        self.cache = IntelligentCache(
            max_size=config.get('cache', {}).get('max_size', 1000),
            default_ttl=config.get('cache', {}).get('default_ttl', 3600),
            policy=CachePolicy(config.get('cache', {}).get('policy', 'hybrid'))
        )
        
        self.circuit_breakers = {}
        self.retry_manager = RetryManager(RetryConfig(
            max_attempts=config.get('retry', {}).get('max_attempts', 3),
            base_delay=config.get('retry', {}).get('base_delay', 1.0)
        ))
        
        self.monitor = PerformanceMonitor(
            alert_threshold=config.get('monitoring', {}).get('alert_threshold', {})
        )
        
        self.resource_limiter = ResourceLimiter(
            max_concurrent=config.get('resources', {}).get('max_concurrent', 100),
            max_memory_mb=config.get('resources', {}).get('max_memory_mb', 1024)
        )
    
    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """회로 차단기 생성/반환"""
        if name not in self.circuit_breakers:
            config = CircuitBreakerConfig()  # 기본 설정 사용
            self.circuit_breakers[name] = CircuitBreaker(name, config)
        
        return self.circuit_breakers[name]
    
    async def optimized_call(self, func: Callable, operation_name: str, 
                           cache_key: Optional[str] = None,
                           use_circuit_breaker: bool = True,
                           use_retry: bool = True,
                           *args, **kwargs) -> Any:
        """최적화된 함수 호출"""
        
        # 1. 캐시 확인
        if cache_key:
            cached_result = await self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        # 2. 리소스 제한 확인
        async with self.resource_limiter.acquire(operation_name):
            
            # 3. 성능 모니터링 시작
            async with self.monitor.measure_time(operation_name):
                
                # 4. 회로 차단기 적용
                if use_circuit_breaker:
                    circuit_breaker = self.get_circuit_breaker(operation_name)
                    
                    async with circuit_breaker.call():
                        # 5. 재시도 메커니즘 적용
                        if use_retry:
                            result = await self.retry_manager.execute_with_retry(
                                func, *args, **kwargs
                            )
                        else:
                            if asyncio.iscoroutinefunction(func):
                                result = await func(*args, **kwargs)
                            else:
                                result = func(*args, **kwargs)
                else:
                    # 회로 차단기 없이 실행
                    if use_retry:
                        result = await self.retry_manager.execute_with_retry(
                            func, *args, **kwargs
                        )
                    else:
                        if asyncio.iscoroutinefunction(func):
                            result = await func(*args, **kwargs)
                        else:
                            result = func(*args, **kwargs)
                
                # 6. 결과 캐싱
                if cache_key and result is not None:
                    await self.cache.set(cache_key, result)
                
                return result
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """종합 성능 통계"""
        return {
            'cache': self.cache.get_stats(),
            'circuit_breakers': {
                name: cb.get_stats() 
                for name, cb in self.circuit_breakers.items()
            },
            'retry': self.retry_manager.get_stats(),
            'monitoring': {
                'system_health': self.monitor.get_system_health(),
                'recent_alerts': self.monitor.alerts[-10:]  # 최근 10개 알림
            },
            'resources': self.resource_limiter.get_resource_stats()
        }

# =====================================
# 7. 에러 핸들링 전략
# =====================================

class ErrorCategory(Enum):
    """에러 카테고리"""
    TRANSIENT = "transient"     # 일시적 에러 (재시도 가능)
    PERSISTENT = "persistent"   # 지속적 에러 (재시도 불가)
    CRITICAL = "critical"       # 치명적 에러 (시스템 중단)
    USER_ERROR = "user_error"   # 사용자 에러

@dataclass
class ErrorHandler:
    """에러 핸들러"""
    category: ErrorCategory
    handler_func: Callable
    retry_allowed: bool = False
    fallback_func: Optional[Callable] = None

class ErrorManagementSystem:
    """에러 관리 시스템"""
    
    def __init__(self):
        self.handlers: Dict[type, ErrorHandler] = {}
        self.error_stats = defaultdict(int)
        self.error_history = deque(maxlen=1000)
    
    def register_handler(self, exception_type: type, handler: ErrorHandler):
        """에러 핸들러 등록"""
        self.handlers[exception_type] = handler
    
    async def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Any:
        """에러 처리"""
        error_type = type(error)
        context = context or {}
        
        # 통계 업데이트
        self.error_stats[error_type.__name__] += 1
        self.error_history.append({
            'error_type': error_type.__name__,
            'message': str(error),
            'timestamp': datetime.now(),
            'context': context
        })
        
        # 핸들러 찾기
        handler = self.handlers.get(error_type)
        if not handler:
            # 기본 핸들러
            handler = self._get_default_handler(error)
        
        # 로깅
        self._log_error(error, handler.category, context)
        
        try:
            # 핸들러 실행
            if asyncio.iscoroutinefunction(handler.handler_func):
                result = await handler.handler_func(error, context)
            else:
                result = handler.handler_func(error, context)
            
            return result
            
        except Exception as handler_error:
            logger.error(f"에러 핸들러 실행 실패: {handler_error}")
            
            # 폴백 함수 시도
            if handler.fallback_func:
                try:
                    if asyncio.iscoroutinefunction(handler.fallback_func):
                        return await handler.fallback_func(error, context)
                    else:
                        return handler.fallback_func(error, context)
                except Exception as fallback_error:
                    logger.error(f"폴백 함수도 실패: {fallback_error}")
            
            # 최종적으로 원본 에러 다시 발생
            raise error
    
    def _get_default_handler(self, error: Exception) -> ErrorHandler:
        """기본 에러 핸들러"""
        # 에러 타입에 따른 기본 분류
        if isinstance(error, (ConnectionError, TimeoutError)):
            category = ErrorCategory.TRANSIENT
        elif isinstance(error, (ValueError, TypeError)):
            category = ErrorCategory.USER_ERROR
        elif isinstance(error, PermissionError):
            category = ErrorCategory.PERSISTENT
        else:
            category = ErrorCategory.CRITICAL
        
        return ErrorHandler(
            category=category,
            handler_func=self._default_error_handler,
            retry_allowed=(category == ErrorCategory.TRANSIENT),
            fallback_func=self._fallback_error_handler
        )
    
    def _log_error(self, error: Exception, category: ErrorCategory, context: Dict[str, Any]):
        """에러 로깅"""
        log_level = {
            ErrorCategory.TRANSIENT: logging.WARNING,
            ErrorCategory.PERSISTENT: logging.ERROR,
            ErrorCategory.CRITICAL: logging.CRITICAL,
            ErrorCategory.USER_ERROR: logging.INFO
        }.get(category, logging.ERROR)
        
        logger.log(
            log_level,
            f"[{category.value.upper()}] {type(error).__name__}: {error}",
            extra={
                'error_type': type(error).__name__,
                'error_category': category.value,
                'context': context,
                'traceback': traceback.format_exc()
            }
        )
    
    def _default_error_handler(self, error: Exception, context: Dict[str, Any]) -> None:
        """기본 에러 핸들러"""
        # 기본적으로는 로깅만 수행
        pass
    
    def _fallback_error_handler(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """폴백 에러 핸들러"""
        return {
            'error': True,
            'message': "시스템 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            'details': str(error) if isinstance(error, (ValueError, TypeError)) else None
        }
    
    def get_error_stats(self) -> Dict[str, Any]:
        """에러 통계"""
        return {
            'error_counts': dict(self.error_stats),
            'recent_errors': list(self.error_history)[-10:],  # 최근 10개
            'total_errors': sum(self.error_stats.values()),
            'error_types': len(self.error_stats)
        }

# =====================================
# 8. 사용 예시 및 통합
# =====================================

async def example_enhanced_rag_with_optimization():
    """최적화된 RAG 시스템 사용 예시"""
    
    # 최적화 매니저 설정
    optimization_config = {
        'cache': {
            'max_size': 2000,
            'default_ttl': 3600,
            'policy': 'hybrid'
        },
        'retry': {
            'max_attempts': 3,
            'base_delay': 1.0
        },
        'monitoring': {
            'alert_threshold': {
                'search_duration': 5.0,
                'generation_duration': 10.0,
                'error_rate': 0.05
            }
        },
        'resources': {
            'max_concurrent': 50,
            'max_memory_mb': 2048
        }
    }
    
    optimizer = OptimizationManager(optimization_config)
    error_manager = ErrorManagementSystem()
    
    # 예시 함수들
    async def expensive_search_operation(query: str) -> List[str]:
        """비용이 큰 검색 작업"""
        await asyncio.sleep(1)  # 시뮬레이션
        return [f"result_{i}" for i in range(5)]
    
    async def expensive_generation_operation(context: str) -> str:
        """비용이 큰 생성 작업"""
        await asyncio.sleep(2)  # 시뮬레이션
        return f"Generated response for: {context}"
    
    # 최적화된 호출
    try:
        # 캐싱을 통한 검색 최적화
        search_results = await optimizer.optimized_call(
            expensive_search_operation,
            operation_name="document_search",
            cache_key="search_query_hash_123",
            use_circuit_breaker=True,
            use_retry=True,
            "사용자 쿼리"
        )
        
        # 리소스 제한을 통한 생성 최적화
        response = await optimizer.optimized_call(
            expensive_generation_operation,
            operation_name="response_generation",
            cache_key="generation_context_456",
            use_circuit_breaker=True,
            use_retry=False,  # 생성은 재시도하지 않음
            str(search_results)
        )
        
        return {
            'response': response,
            'search_results': search_results,
            'optimization_stats': optimizer.get_comprehensive_stats()
        }
        
    except Exception as e:
        # 에러 관리 시스템 통합
        error_result = await error_manager.handle_error(e, {
            'operation': 'enhanced_rag_request',
            'user_query': '사용자 쿼리'
        })
        return error_result

# 설정 기반 초기화
def create_optimization_manager_from_config(config: Dict[str, Any]) -> OptimizationManager:
    """설정에서 최적화 매니저 생성"""
    monitoring_config = config.get('monitoring', {})
    
    optimization_config = {
        'cache': {
            'max_size': monitoring_config.get('cache_max_size', 1000),
            'default_ttl': monitoring_config.get('cache_ttl', 3600),
            'policy': monitoring_config.get('cache_policy', 'hybrid')
        },
        'retry': {
            'max_attempts': monitoring_config.get('retry_max_attempts', 3),
            'base_delay': monitoring_config.get('retry_base_delay', 1.0)
        },
        'monitoring': {
            'alert_threshold': {
                'search_duration': 5.0,
                'generation_duration': 10.0,
                'error_rate': 0.05
            }
        },
        'resources': {
            'max_concurrent': monitoring_config.get('max_concurrent_requests', 50),
            'max_memory_mb': monitoring_config.get('max_memory_mb', 2048)
        }
    }
    
    return OptimizationManager(optimization_config)

if __name__ == "__main__":
    # 성능 최적화 시스템 테스트
    asyncio.run(example_enhanced_rag_with_optimization())