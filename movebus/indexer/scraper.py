"""Scraper para coleta de dados de linhas de ônibus do site onibus.info.

Usa Playwright para renderizar a SPA, clicar na aba Linhas e extrair
os dados de cada linha de Joinville/SC.
"""

import logging
import re
import time
from typing import Optional

from movebus.models import BusLineRaw

logger = logging.getLogger(__name__)

_BASE_URL = "https://onibus.info"
_HOME_URL = "https://onibus.info/"
_PAGE_TIMEOUT = 60000   # ms
_ITEM_TIMEOUT = 20000   # ms


class BusScraper:
    """Coleta dados de linhas de ônibus de Joinville a partir do onibus.info."""

    def __init__(self, headless: bool = False) -> None:
        self._headless = headless

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrape_all_lines(self) -> list[BusLineRaw]:
        """Coleta todas as linhas de ônibus de Joinville."""
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            logger.error(
                "Playwright não instalado. Execute: "
                "pip install playwright && python -m playwright install chromium"
            )
            return []

        line_urls: list[str] = []
        results: list[BusLineRaw] = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self._headless)
            page = browser.new_page()

            try:
                # Navega para a home e clica em Linhas (igual ao debug que funcionou)
                logger.info("Acessando home do onibus.info...")
                page.goto(_HOME_URL, timeout=_PAGE_TIMEOUT)
                time.sleep(3)

                # Clica na aba Linhas
                for el in page.query_selector_all("a, button, li"):
                    if el.inner_text().strip() == "Linhas":
                        el.click()
                        break

                # Aguarda os links aparecerem (polling manual)
                logger.info("Aguardando links de linhas...")
                for _ in range(30):
                    links = page.query_selector_all("a[href^='/linhas/']")
                    if len(links) > 10:
                        break
                    time.sleep(1)
                time.sleep(2)

                # Extrai todas as URLs de linhas
                line_urls = self._extract_line_urls(page)
                logger.info("%d URLs de linhas encontradas.", len(line_urls))

                # Coleta dados de cada linha
                for i, url in enumerate(line_urls):
                    logger.info("Coletando linha %d/%d: %s", i + 1, len(line_urls), url)
                    bus_line = self._scrape_line_page(page, url)
                    if bus_line is not None:
                        results.append(bus_line)

            except Exception:
                logger.exception("Erro ao fazer scraping da listagem.")
            finally:
                browser.close()

        logger.info(
            "Scraping concluído: %d/%d linhas coletadas.",
            len(results),
            len(line_urls),
        )
        return results

    def scrape_line(self, url: str) -> Optional[BusLineRaw]:
        """Coleta dados de uma linha específica pela URL."""
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            logger.error("Playwright não instalado.")
            return None

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self._headless)
            page = browser.new_page()
            try:
                result = self._scrape_line_page(page, url)
            finally:
                browser.close()
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_line_urls(self, page) -> list[str]:
        """Extrai URLs únicas de linhas da página já carregada."""
        anchors = page.query_selector_all("a[href^='/linhas/']")
        seen: set[str] = set()
        urls: list[str] = []

        for anchor in anchors:
            href = anchor.get_attribute("href") or ""
            # Só aceita /linhas/<id> — ignora /linhas sem id
            if re.match(r"^/linhas/\w+$", href):
                full_url = _BASE_URL + href
                if full_url not in seen:
                    seen.add(full_url)
                    urls.append(full_url)

        return urls

    def _scrape_line_page(self, page, url: str) -> Optional[BusLineRaw]:
        """Navega para a página de uma linha e extrai os dados."""
        try:
            page.goto(url, timeout=_PAGE_TIMEOUT)
            # Aguarda o nome da linha aparecer
            page.wait_for_selector(
                "h1, h2, [class*='route'], [class*='linha'], [class*='title']",
                timeout=_ITEM_TIMEOUT,
            )
            time.sleep(1)

            html = page.content()

            # Nome da linha
            line_name = self._extract_text(
                page,
                "h1, h2, [class*='route-name'], [class*='linha-nome']",
            )
            if not line_name:
                # Fallback: pega do título da página
                title = page.title()
                line_name = title.split("|")[0].strip()

            line_number = self._extract_line_number_from_url(url)
            itinerary = self._extract_itinerary(page, line_name)
            schedules, stops = self._extract_schedules_and_stops(page)

            logger.debug("Linha coletada: %s — %d horários, %d paradas",
                         line_name, len(schedules), len(stops))

            return BusLineRaw(
                url=url,
                line_name=line_name,
                line_number=line_number,
                itinerary=itinerary,
                schedules=schedules,
                stops=stops,
                raw_html=html,
            )
        except Exception:
            logger.exception("Erro ao coletar linha %s", url)
            return None

    @staticmethod
    def _extract_text(page, selector: str) -> str:
        el = page.query_selector(selector)
        return el.inner_text().strip() if el else ""

    @staticmethod
    def _extract_line_number_from_url(url: str) -> str:
        """Extrai o número da linha a partir da URL (/linhas/0040 → '0040')."""
        match = re.search(r"/linhas/(\w+)$", url)
        return match.group(1) if match else ""

    @staticmethod
    def _extract_itinerary(page, line_name: str) -> str:
        """Extrai o itinerário a partir do texto da página."""
        # O site mostra "Terminal A → Terminal B" no topo da página de linha
        for selector in [
            ".onibus-route__description", "[class*='itinerar']",
            "[class*='subtitle']", "[class*='rota']",
        ]:
            el = page.query_selector(selector)
            if el:
                text = el.inner_text().strip()
                if len(text) > 5:
                    return text
        return line_name

    @staticmethod
    def _extract_schedules_and_stops(page) -> tuple[list[str], list[str]]:
        """Extrai horários e paradas do corpo da página.

        O site renderiza cada parada como: HH:MM <ícone> <nome da parada>
        Extrai ambos em uma única passagem pelo texto.
        """
        body_text = page.inner_text("body")

        # Padrão: horário seguido de nome de parada na próxima linha
        # Ex: "22:15\nrss_feed\nTerminal Norte\nPlataforma 1, Box 5"
        lines = body_text.split("\n")

        schedules: list[str] = []
        stops: list[str] = []
        seen_schedules: set[str] = set()
        seen_stops: set[str] = set()

        # Itens de navegação a ignorar
        nav_items = {
            "notícias", "linhas", "meu local", "como chegar",
            "buscar", "contato", "início", "noticias", "topo",
            "rss_feed", "keyboard_arrow_down", "arrow_upward",
            "itinerário", "horários", "alertas", "dias úteis",
            "sábados", "domingos", "directions_bus", "place",
            "arrow_forward_ios", "circular",
        }

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Detecta horário
            if re.match(r"^\d{1,2}:\d{2}$", line):
                if line not in seen_schedules:
                    seen_schedules.add(line)
                    schedules.append(line)
                # A parada vem após o ícone rss_feed (1-2 linhas depois)
                j = i + 1
                while j < min(i + 4, len(lines)):
                    candidate = lines[j].strip()
                    if (candidate
                            and candidate.lower() not in nav_items
                            and not re.match(r"^\d{1,2}:\d{2}$", candidate)
                            and len(candidate) > 3):
                        stop_text = candidate
                        # Concatena linha seguinte se for complemento (ex: "Plataforma 1, Box 5")
                        if j + 1 < len(lines):
                            next_line = lines[j + 1].strip()
                            if (next_line
                                    and next_line.lower() not in nav_items
                                    and not re.match(r"^\d{1,2}:\d{2}$", next_line)
                                    and len(next_line) > 2
                                    and not next_line.startswith("keyboard")):
                                stop_text = f"{candidate}, {next_line}"
                        if stop_text not in seen_stops:
                            seen_stops.add(stop_text)
                            stops.append(stop_text)
                        break
                    j += 1
            i += 1

        return schedules, stops

    @staticmethod
    def _extract_schedules(page) -> list[str]:
        schedules, _ = BusScraper._extract_schedules_and_stops(page)
        return schedules

    @staticmethod
    def _extract_stops(page) -> list[str]:
        _, stops = BusScraper._extract_schedules_and_stops(page)
        return stops
