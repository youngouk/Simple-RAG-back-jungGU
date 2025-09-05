# 프롬프트 관리 시스템 통합 가이드

## 개요
이 가이드는 프론트엔드 개발자가 RAG 챗봇의 시스템 프롬프트를 관리할 수 있는 기능을 통합하는 방법을 설명합니다.

## 시스템 아키텍처

### 백엔드 구현 완료 사항
1. **데이터 모델** (`/app/models/prompts.py`)
   - 프롬프트 CRUD 모델 정의
   - 카테고리 기반 프롬프트 관리

2. **프롬프트 매니저** (`/app/modules/prompt_manager.py`)
   - 파일 기반 프롬프트 저장소 (JSON)
   - 기본 프롬프트 자동 생성
   - 프롬프트 가져오기/내보내기 기능

3. **API 엔드포인트** (`/app/api/prompts.py`)
   - RESTful API 제공
   - 프롬프트 CRUD 작업 지원

4. **생성 모듈 통합** (`/app/modules/generation.py`)
   - 동적 프롬프트 로딩
   - 스타일 기반 프롬프트 선택

## API 엔드포인트는 기존 설정 사용


### 1. 프롬프트 목록 조회
```http
GET /api/prompts?category=system&is_active=true&page=1&page_size=50
```

**응답 예시:**
```json
{
  "prompts": [
    {
      "id": "uuid-string",
      "name": "system",
      "content": "당신은 유저의 질문을 분석/판단하고...",
      "description": "기본 시스템 프롬프트",
      "category": "system",
      "is_active": true,
      "metadata": {},
      "created_at": "2024-01-01T00:00:00",
      "updated_at": "2024-01-01T00:00:00"
    }
  ],
  "total": 5,
  "page": 1,
  "page_size": 50
}
```

### 2. 특정 프롬프트 조회 (ID)
```http
GET /api/prompts/{prompt_id}
```

### 3. 이름으로 프롬프트 조회
```http
GET /api/prompts/by-name/{name}
```

### 4. 새 프롬프트 생성
```http
POST /api/prompts
Content-Type: application/json

{
  "name": "custom_prompt",
  "content": "프롬프트 내용...",
  "description": "커스텀 프롬프트",
  "category": "custom",
  "is_active": true
}
```

### 5. 프롬프트 수정
```http
PUT /api/prompts/{prompt_id}
Content-Type: application/json

{
  "content": "수정된 프롬프트 내용...",
  "is_active": false
}
```

### 6. 프롬프트 삭제
```http
DELETE /api/prompts/{prompt_id}
```

### 7. 프롬프트 내보내기 (백업)
```http
GET /api/prompts/export/all
```

### 8. 프롬프트 가져오기 (복원)
```http
POST /api/prompts/import?overwrite=false
Content-Type: application/json

{
  "prompts": [...],
  "exported_at": "2024-01-01T00:00:00",
  "total": 5
}
```

## 프롬프트 카테고리 및 스타일

### 카테고리
- `system`: 시스템 기본 프롬프트
- `style`: 답변 스타일 프롬프트
- `custom`: 사용자 정의 프롬프트

### 지원되는 스타일 (name 필드)
- `system`: 기본 시스템 프롬프트
- `detailed`: 자세한 답변
- `concise`: 간결한 답변
- `professional`: 전문적 답변
- `educational`: 교육적 답변

## 프론트엔드 구현 가이드

### 1. 프롬프트 관리 UI 컴포넌트

```jsx
// PromptManager.jsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || '';

const PromptManager = () => {
  const [prompts, setPrompts] = useState([]);
  const [selectedPrompt, setSelectedPrompt] = useState(null);
  const [editMode, setEditMode] = useState(false);
  
  // 프롬프트 목록 로드
  const loadPrompts = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/prompts`);
      setPrompts(response.data.prompts);
    } catch (error) {
      console.error('Failed to load prompts:', error);
    }
  };
  
  // 프롬프트 저장
  const savePrompt = async (promptData) => {
    try {
      if (editMode && selectedPrompt?.id) {
        // 수정
        await axios.put(
          `${API_BASE_URL}/api/prompts/${selectedPrompt.id}`,
          promptData
        );
      } else {
        // 생성
        await axios.post(`${API_BASE_URL}/api/prompts`, promptData);
      }
      loadPrompts();
    } catch (error) {
      console.error('Failed to save prompt:', error);
    }
  };
  
  // 프롬프트 삭제
  const deletePrompt = async (promptId) => {
    try {
      await axios.delete(`${API_BASE_URL}/api/prompts/${promptId}`);
      loadPrompts();
    } catch (error) {
      console.error('Failed to delete prompt:', error);
    }
  };
  
  useEffect(() => {
    loadPrompts();
  }, []);
  
  // UI 렌더링...
};
```

### 2. 어드민 대시보드 통합

어드민 페이지에 프롬프트 관리 탭 추가:

```jsx
// AdminDashboard.jsx
import PromptManager from './PromptManager';

const AdminDashboard = () => {
  return (
    <Tabs>
      <Tab label="대시보드">...</Tab>
      <Tab label="프롬프트 관리">
        <PromptManager />
      </Tab>
      <Tab label="문서 관리">...</Tab>
    </Tabs>
  );
};
```

### 3. 채팅 인터페이스 스타일 선택

채팅 UI에서 답변 스타일 선택 기능:

```jsx
// ChatInterface.jsx
const ChatInterface = () => {
  const [style, setStyle] = useState('system');
  
  const sendMessage = async (message) => {
    const response = await axios.post(`${API_BASE_URL}/api/chat`, {
      message,
      style, // 선택된 스타일 전달
      session_id: sessionId
    });
    // 응답 처리...
  };
  
  return (
    <div>
      <Select value={style} onChange={(e) => setStyle(e.target.value)}>
        <Option value="system">기본</Option>
        <Option value="detailed">자세히</Option>
        <Option value="concise">간결하게</Option>
        <Option value="professional">전문적으로</Option>
        <Option value="educational">교육적으로</Option>
      </Select>
      {/* 채팅 UI */}
    </div>
  );
};
```

## 프롬프트 작성 가이드

### 좋은 프롬프트의 특징
1. **명확한 역할 정의**: AI의 역할과 전문성을 명시
2. **구체적인 지시사항**: 답변 방식과 형식 지정
3. **제약 사항 명시**: 하지 말아야 할 것들 정의
4. **예시 포함**: 원하는 답변 형식의 예시 제공

### 프롬프트 템플릿 예시

```text
# 전문가 프롬프트
당신은 [분야]의 전문가입니다.
다음 지침을 따라 답변해주세요:
1. [지침 1]
2. [지침 2]
3. [지침 3]

제공된 문서를 기반으로 답변하되, 없는 내용은 추측하지 마세요.
전문 용어는 쉽게 설명해주세요.
```

## 보안 고려사항

1. **인증/권한**: 프롬프트 관리는 관리자만 접근 가능하도록 구현
2. **입력 검증**: XSS, SQL Injection 방지를 위한 입력 검증
3. **백업**: 정기적인 프롬프트 백업 (export 기능 활용)
4. **버전 관리**: 프롬프트 변경 이력 관리 고려

## 테스트 시나리오

### 1. 기본 CRUD 테스트
- [ ] 프롬프트 목록 조회
- [ ] 새 프롬프트 생성
- [ ] 프롬프트 수정
- [ ] 프롬프트 삭제
- [ ] 프롬프트 활성화/비활성화

### 2. 통합 테스트
- [ ] 채팅에서 스타일 변경 시 응답 변화 확인
- [ ] 프롬프트 수정 후 즉시 반영 확인
- [ ] 비활성화된 프롬프트 사용 시 폴백 확인

### 3. 성능 테스트
- [ ] 다수 프롬프트 로딩 속도
- [ ] 프롬프트 전환 응답 시간

## 트러블슈팅

### 문제: 프롬프트가 적용되지 않음
**해결방법:**
1. 프롬프트 활성화 상태 확인
2. 프롬프트 이름과 스타일 매핑 확인
3. generation.py 로그 확인

### 문제: 프롬프트 저장 실패
**해결방법:**
1. data/prompts 디렉토리 권한 확인
2. 프롬프트 이름 중복 확인
3. JSON 형식 유효성 확인

## 추가 개발 제안

### 1. 프롬프트 버전 관리
- 변경 이력 저장
- 롤백 기능
- 비교 기능

### 2. A/B 테스팅
- 프롬프트 성능 비교
- 사용자 만족도 측정
- 자동 최적화

### 3. 프롬프트 템플릿
- 사전 정의된 템플릿 제공
- 변수 치환 기능
- 조건부 프롬프트

## 연락처 및 지원

백엔드 관련 질문이나 이슈가 있으시면 아래 정보를 참고하세요:

- 로그 위치: `./logs/app.log`
- 프롬프트 저장 위치: `./data/prompts/prompts.json`

## 부록: API 응답 코드

| 코드 | 설명 |
|------|------|
| 200 | 성공 |
| 201 | 생성 성공 |
| 400 | 잘못된 요청 (중복 이름 등) |
| 404 | 프롬프트를 찾을 수 없음 |
| 500 | 서버 오류 |