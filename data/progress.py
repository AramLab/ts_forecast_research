"""
data/progress.py
Прогресс-бар с оценкой оставшегося времени для длинных прогонов.
"""
import time
import sys


class ProgressTracker:
    """
    Отслеживает прогресс обработки N рядов.
    Показывает: [██████░░░░] 60/100 (60%) | ETA: 3м 20с | avg: 5.2с/ряд
    """
    def __init__(self, total: int, dataset_name: str = "", bar_width: int = 30):
        self.total = total
        self.dataset_name = dataset_name
        self.bar_width = bar_width
        self.done = 0
        self.start_time = time.time()
        self.times = []          # время каждого ряда
        self.failed = 0
        self.skipped = 0
        self._last_print = 0

    def update(self, elapsed_sec: float, success: bool = True, skipped: bool = False):
        """Вызывать после каждого обработанного ряда."""
        self.done += 1
        if skipped:
            self.skipped += 1
        elif not success:
            self.failed += 1
        else:
            self.times.append(elapsed_sec)
        self._print()

    def _eta_str(self) -> str:
        remaining = self.total - self.done
        if not self.times or remaining == 0:
            return "—"
        avg = sum(self.times[-20:]) / len(self.times[-20:])  # скользящее среднее
        eta_sec = avg * remaining
        if eta_sec < 60:
            return f"{int(eta_sec)}с"
        elif eta_sec < 3600:
            return f"{int(eta_sec // 60)}м {int(eta_sec % 60)}с"
        else:
            h = int(eta_sec // 3600)
            m = int((eta_sec % 3600) // 60)
            return f"{h}ч {m}м"

    def _avg_str(self) -> str:
        if not self.times:
            return "—"
        avg = sum(self.times[-20:]) / len(self.times[-20:])
        return f"{avg:.1f}с/ряд"

    def _print(self):
        now = time.time()
        # Не чаще раза в секунду
        if now - self._last_print < 1.0 and self.done < self.total:
            return
        self._last_print = now

        pct = self.done / self.total
        filled = int(self.bar_width * pct)
        bar = "█" * filled + "░" * (self.bar_width - filled)

        elapsed = now - self.start_time
        elapsed_str = f"{int(elapsed // 60)}м {int(elapsed % 60)}с" if elapsed >= 60 else f"{int(elapsed)}с"

        prefix = f"[{self.dataset_name}] " if self.dataset_name else ""
        line = (
            f"\r  {prefix}[{bar}] {self.done}/{self.total} ({pct:.0%})"
            f" | ETA: {self._eta_str()}"
            f" | {self._avg_str()}"
            f" | elapsed: {elapsed_str}"
        )
        if self.failed:
            line += f" | ❌{self.failed}"
        if self.skipped:
            line += f" | ⚠{self.skipped}"

        sys.stdout.write(line)
        sys.stdout.flush()

        if self.done >= self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def summary(self):
        elapsed = time.time() - self.start_time
        processed = self.done - self.skipped - self.failed
        avg = sum(self.times) / len(self.times) if self.times else 0
        print(f"\n  Итог [{self.dataset_name}]:")
        print(f"    Всего:     {self.total}")
        print(f"    Обработано:{processed}")
        print(f"    Пропущено: {self.skipped}")
        print(f"    Ошибки:    {self.failed}")
        elapsed_str = f"{int(elapsed // 60)}м {int(elapsed % 60)}с" if elapsed >= 60 else f"{int(elapsed)}с"
        print(f"    Время:     {elapsed_str}")
        print(f"    Среднее:   {avg:.1f}с/ряд")
