import re
from bs4 import BeautifulSoup

def parse_html_prompt(input_str):
    soup = BeautifulSoup(input_str, "html.parser")
    
    # 处理 <p></p> 内的内容
    p_content = soup.find("p").decode_contents().replace("<br>", "\n")
    p_content = re.sub(r'<span[^>]*>(.*?)</span>', r'<\1>', p_content)
    template = p_content.strip().replace(' <br/>', '').replace(' ', '').replace('<br/>', '')
    
    # 解析 component-item
    components = {}
    for item in soup.find_all("div", class_="component-item"):
        key_span = item.find("div", class_="component-key").find("span")
        key = key_span.get_text(strip=True) if key_span else ""
        value_div = item.find("div", class_="component-value")
        value_content = value_div.decode_contents()
        value_content = re.sub(r'<span[^>]*>(.*?)</span>', r'{\1}', value_content)
        components[key] = value_content.strip().replace(' <br/>', '').replace('<br/>', '')
    
    # 解析 self-info-item
    self_prompt = {}
    for item in soup.find_all("div", class_="self-info-item"):
        key_span = item.find("div", class_="component-key").find("span")
        key = key_span.get_text(strip=True) if key_span else ""
        value_div = item.find("div", class_="component-value")
        value = value_div.get_text(strip=True) if value_div else ""
        self_prompt[key] = value.replace(' <br/>', '').replace('<br/>', '')
    
    return {
        'template': template,
        'components': components,
        'self_prompt': self_prompt
    }

#print(parse_html(info))