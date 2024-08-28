from datetime import datetime
from langchain_upstage import UpstageGroundednessCheck
from bs4 import BeautifulSoup
from markdownify import markdownify as md

current_date = datetime.now().strftime("%Y년 %m월 %d일")

def perform_groundedness_check(context, answer) -> str:
    groundedness_check = UpstageGroundednessCheck()
    response = groundedness_check.invoke({
        "context": context,
        "answer": answer
    })

    # 검증완료 (grounded)
    # 검증실패 (notGrounded)
    # 검증불가 (notSure)

    return response

grounded_result_mapping = {
    "grounded": ("검증완료", "green"),
    "notGrounded": ("검증실패", "red"),
    "notSure": ("검증불가", "orange")
}

def html_to_markdown(html_content):
    # BeautifulSoup을 사용하여 HTML 파싱
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 불필요한 요소 제거 (예: 스크립트, 스타일)
    for script in soup(["script", "style"]):
        script.decompose()
    
    # 특정 요소 전처리 (필요한 경우)
    # 예: 특정 클래스를 가진 div를 제거하거나 수정
    # for div in soup.find_all("div", class_="unwanted-class"):
    #     div.decompose()
    
    # markdownify를 사용하여 HTML을 Markdown으로 변환
    markdown = md(str(soup), heading_style="ATX", strip=['a'])
    
    return markdown.strip()