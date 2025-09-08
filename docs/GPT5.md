GPT-5 API 기본 안내서
제공된 API 문서를 바탕으로 GPT-5 모델 호출을 위한 기본 안내서를 다음과 같이 정리했습니다.

1. GPT-5 모델 종류 및 선택
GPT-5 시리즈는 세 가지 모델로 제공되며, 작업의 복잡성과 요구 사항에 따라 선택할 수 있습니다.

gpt-5: 복잡한 추론, 광범위한 세상 지식, 코드 중심적이거나 여러 단계의 에이전트 작업에 가장 적합합니다.

gpt-5-mini: 비용에 최적화된 추론 및 채팅 모델로, 속도, 비용, 능력의 균형을 맞춥니다.

gpt-5-nano: 간단한 지시를 따르거나 분류와 같은 처리량이 많은 작업에 적합합니다.

2. 빠른 응답을 위한 옵션
기본적으로 GPT-5는 중간 길이의 사고 과정을 거친 후 응답하지만, 더 빠른 응답이 필요한 경우 다음과 같이 reasoning과 text 매개변수를 조정할 수 있습니다.

Python

from openai import OpenAI

client = OpenAI()

result = client.responses.create(
    model="gpt-5",
    input="코드에 대한 하이쿠를 써줘.",
    reasoning={ "effort": "low" },  # 추론 노력 최소화
    text={ "verbosity": "low" },   # 응답 상세도 낮춤
)

print(result.output_text)
3. 모델 이름 참조
API 호출 시에는 다음 표를 참고하여 정확한 모델 이름을 사용해야 합니다.

시스템 카드 이름	API 별칭 (사용할 이름)
gpt-5-thinking	gpt-5
gpt-5-thinking-mini	gpt-5-mini
gpt-5-thinking-nano	gpt-5-nano
gpt-5-main	gpt-5-chat-latest
gpt-5-main-mini	[API로 사용 불가]

Sheets로 내보내기
4. 새로운 API 기능
GPT-5와 함께 개발자에게 더 많은 제어 기능과 유연성을 제공하는 새로운 기능들이 도입되었습니다.

응답 상세도 조절 (Verbosity Control): text 매개변수를 통해 응답의 상세함을 조절합니다.

최소 추론 노력 옵션 (Minimal Reasoning Effort): reasoning 매개변수를 통해 응답 속도를 높입니다.

사용자 정의 도구 (Custom Tools)

허용된 도구 목록 (Allowed Tools List)