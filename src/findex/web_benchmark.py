import argparse
import asyncio
import statistics
import time

import httpx

ROUTES: dict[str, str] = {
    "def  (/search)": "/search?q={q}&k=10",
    "async def + блокуючий код": "/_bench/search-async?q={q}",
    "async def + to_thread": "/_bench/search-to-thread?q={q}",
}


async def one_round(
    client: httpx.AsyncClient, url: str, n: int
) -> tuple[float, list[float], float]:
    """n запитів одночасно: (загальний час, час кожного запиту, час /health), у секундах.

    /health — «сторонній» запит, який приходить, коли пошуки вже в роботі. Якщо маршрут блокує
    цикл подій, навіть цей найлегший запит стоїть у черзі за всіма пошуками.
    """

    async def timed_get() -> float:
        started = time.perf_counter()
        response = await client.get(url)
        response.raise_for_status()
        return time.perf_counter() - started

    async def health_probe() -> float:
        await asyncio.sleep(0.2)
        return await timed_health()

    async def timed_health() -> float:
        started = time.perf_counter()
        (await client.get("/health")).raise_for_status()
        return time.perf_counter() - started

    started = time.perf_counter()
    *latencies, health = await asyncio.gather(*(timed_get() for _ in range(n)), health_probe())
    return time.perf_counter() - started, latencies, health


async def main_async(base: str, query: str, n: int, repeat: int) -> None:
    print(
        f"| Маршрут | {n} запитів одночасно, загалом (с) | p50 запиту (с) | найповільніший (с) | /health під навантаженням (с) |"
    )
    print("|---|---|---|---|---|")
    async with httpx.AsyncClient(base_url=base, timeout=120) as client:
        for name, template in ROUTES.items():
            url = template.format(q=query)
            await one_round(client, url, 2)  # розігрів: перший запит прогріває кеші
            rounds = [await one_round(client, url, n) for _ in range(repeat)]
            wall = statistics.median(total for total, _, _ in rounds)
            lat = [x for _, latencies, _ in rounds for x in latencies]
            health = statistics.median(h for _, _, h in rounds)
            print(
                f"| {name} | {wall:.2f} | {statistics.median(lat):.2f} | {max(lat):.2f} "
                f"| {health:.2f} |"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--query", default="chess")
    parser.add_argument("-n", type=int, default=20, help="скільки запитів одночасно")
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(main_async(args.base, args.query, args.n, args.repeat))


if __name__ == "__main__":
    main()
