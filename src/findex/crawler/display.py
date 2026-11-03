"""Живий екран краулера (rich): лічильники читаються з CrawlStats, яку оновлює краулер."""

from rich.table import Table

from findex.crawler.crawl import CrawlStats


def render_stats(stats: CrawlStats, max_pages: int) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column(justify="right")
    table.add_row("Зібрано сторінок", f"{stats.fetched} / {max_pages}")
    table.add_row("Сторінок за секунду", f"{stats.pages_per_second:.1f}")
    table.add_row("Запитів чекають відповіді", str(stats.in_flight))
    table.add_row("Помилок", str(stats.errors))
    table.add_row("Повторів (429/5xx/таймаут)", str(stats.retries))
    table.add_row("Заборонено robots.txt", str(stats.robots_skipped))
    table.add_row("Черга", f"{stats.queue_depth} (бачили {stats.seen})")
    table.add_row("Минуло", f"{stats.elapsed:.1f} с")
    return table
