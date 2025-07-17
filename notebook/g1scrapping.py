import requests
from bs4 import BeautifulSoup

def get_html_from_url(url):
    """
    Captura o conteúdo HTML de uma URL e retorna um objeto BeautifulSoup.

    Args:
        url (str): O link do site para capturar o HTML.

    Returns:
        BeautifulSoup object: Um objeto BeautifulSoup contendo o HTML parseado,
                              ou None se houver um erro.
    """
    try:
        # Faz a requisição HTTP GET para a URL
        response = requests.get(url)

        # Verifica se a requisição foi bem sucedida (código de status 200)
        response.raise_for_status()

        # Cria um objeto BeautifulSoup para parsear o conteúdo HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        return soup

    except requests.exceptions.RequestException as e:
        print(f"Erro ao acessar a URL: {e}")
        return None
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
        return None

def page_html(url_do_site):
    html_soup = get_html_from_url(url_do_site)
    
    if html_soup:
        print("HTML capturado com sucesso!")
        # Você pode agora trabalhar com o objeto 'html_soup' para extrair informações.
        # Por exemplo, para imprimir o título da página:
        if html_soup.title:
            print(f"Título da página: {html_soup.title.string}")
        else:
            print("Título da página não encontrado.")
    
        # Para imprimir uma parte do HTML (as primeiras 500 caracteres, por exemplo)
        # print("\nPrimeiras 500 caracteres do HTML:")
        # print(str(html_soup)[:500])
    else:
        print("Não foi possível capturar o HTML do site.")

    return html_soup


def html_to_array(html, tag, class_tag):
    if class_tag:
        specific_div = html.find_all(tag, class_=class_tag)
    else:
        specific_div = html.find_all(tag)
        
    list_info = []
    for p in specific_div:
        actual_text = p.text.strip()
        if actual_text:
            list_info.append(actual_text)
            
    return list_info

def get_g1globo_page(link):
    noticia = get_html_from_url(link)

    todos_paragrafos = html_to_array(noticia, "p", "")
        
    specific_div = noticia.find_all('div', class_='title')
    titulo = ""
    for p in specific_div:
        titulo = titulo + f"{titulo} {p.text} \n"
        
    titulo = titulo.strip()

    specific_div = noticia.find_all('h2', class_='content-head__subtitle')
    subtitulo = ""
    for p in specific_div:
        subtitulo = subtitulo + f"{subtitulo} {p.text} \n"
        
    subtitulo = subtitulo.strip()

    resumo = html_to_array(noticia, "li", "mc-summary-card__item")
    
    return {
        "titulo": titulo,
        "subtitulo": subtitulo,
        "resumo": resumo,
        "link": link,
        "todos_paragrafos": todos_paragrafos
    }


url_do_site = "https://www.globo.com/"  # Substitua pela URL do site que você quer capturar
html_soup = page_html(url_do_site)


todos_paragrafos = html_soup.find_all('a')
link_lists = []

for p in todos_paragrafos: # pegar todos links da página inicial
    actual_link = p.get('href')
    if ("/noticia" in actual_link) and (".ghtml" in actual_link):
        link_lists.append(p.get('href'))
        
all_info_dict = []
for link in link_lists:
    if "g1.globo" in link:
        all_info_dict.append(get_g1globo_page(link))