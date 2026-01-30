#!/usr/bin/python3
"""
Плавающее окно для записи и распознавания речи с микрофона.

Архитектура приложения специально осознанно выбрана как God-file
"""

from __future__ import annotations
import sys
import wave
import time
import numpy as np
import sounddevice as sd
import onnx_asr
import threading
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import AutoProcessor, pipeline as hf_pipeline
import signal
import json
import os
import subprocess
import shutil
import shlex
import httpx
import gi
from dotenv import load_dotenv

# Загружаем переменные из .env файла, если он существует
load_dotenv()

from enum import Enum
from typing import Callable, Optional, Protocol, Union, List
from dataclasses import dataclass, replace

gi.require_version("Gtk", "3.0")
gi.require_version("GtkLayerShell", "0.1")
gi.require_version("Gdk", "3.0")
from gi.repository import Gtk, GtkLayerShell, GLib, Gdk


# ============================================================================
# REDUX АРХИТЕКТУРА - УПРАВЛЕНИЕ СОСТОЯНИЕМ
# ============================================================================


class Phase(Enum):
    """Фазы приложения"""

    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"
    POST_PROCESSING = "post_processing"
    RESTARTING = "restarting"


@dataclass(frozen=True)
class State:
    """
    Неизменяемое состояние приложения.

    Все изменения состояния проходят через reducer, который возвращает новый экземпляр State.
    Это обеспечивает предсказуемые переходы состояний и устраняет race conditions.
    """

    # Текущая фаза
    phase: Phase = Phase.IDLE

    # Настройки (изменяемые в runtime)
    llm_enabled: bool = True  # Use alternative ASR model (Whisper) instead of gigaam
    auto_paste: bool = True
    copy_method: str = "clipboard"  # "clipboard" | "primary"
    smart_text_processing: bool = False
    smart_short_phrase_words: int = 3

    # Данные
    recognized_text: Optional[str] = None
    processed_text: Optional[str] = None
    error: Optional[str] = None

    # Связанное с интерфейсом
    current_monitor_name: Optional[str] = None
    rel_x: float = 0.5  # Относительная позиция центра (0.0 - 1.0)
    rel_y: float = 0.1  # Относительная позиция центра (0.0 - 1.0)


# --- UI события ---
@dataclass(frozen=True)
class UIStart:
    """Пользователь нажал кнопку старт/запись"""

    pass


@dataclass(frozen=True)
class UIStop:
    """Пользователь нажал кнопку стоп"""

    pass


@dataclass(frozen=True)
class UIRestart:
    """Пользователь нажал кнопку перезапуска во время записи"""

    pass


@dataclass(frozen=True)
class UIToggleAltModel:
    """Пользователь переключил альтернативную ASR модель"""

    pass


# --- Результаты async операций ---
@dataclass(frozen=True)
class ASRDone:
    """ASR (распознавание речи) завершено"""

    text: Optional[str]
    error: Optional[str] = None


@dataclass(frozen=True)
class LLMDone:
    """LLM пост-обработка завершена"""

    text: Optional[str]
    error: Optional[str] = None


@dataclass(frozen=True)
class RestartDone:
    """Перезапуск записи завершён"""

    success: bool
    error: Optional[str] = None


# --- Системные события ---
@dataclass(frozen=True)
class MonitorChanged:
    """Конфигурация монитора изменилась или окно перемещено на новый монитор"""

    monitor_name: Optional[str]
    rel_x: Optional[float] = None
    rel_y: Optional[float] = None


@dataclass(frozen=True)
class WindowPositionChanged:
    """Относительная позиция окна изменилась"""

    rel_x: float
    rel_y: float
    is_manual: bool = False


# Union тип для всех действий
Action = Union[
    UIStart,
    UIStop,
    UIRestart,
    UIToggleAltModel,
    ASRDone,
    LLMDone,
    RestartDone,
    MonitorChanged,
    WindowPositionChanged,
]


class Reducer:
    """
    Чистый reducer - обрабатывает переходы состояний без побочных эффектов.

    Каждый обработчик:
    1. Проверяет, разрешён ли переход
    2. Возвращает новый State (или тот же state, если переход невалиден)
    3. Никогда не выполняет I/O или async операции
    """

    @staticmethod
    def handle_ui_start(state: State, action: UIStart) -> State:
        """Пользователь хочет начать запись"""
        if state.phase != Phase.IDLE:
            return state

        return replace(
            state,
            phase=Phase.RECORDING,
            error=None,
            recognized_text=None,
            processed_text=None,
        )

    @staticmethod
    def handle_ui_stop(state: State, action: UIStop) -> State:
        """Пользователь хочет остановить запись и распознать"""
        if state.phase != Phase.RECORDING:
            return state

        return replace(state, phase=Phase.PROCESSING, error=None)

    @staticmethod
    def handle_ui_restart(state: State, action: UIRestart) -> State:
        """Пользователь хочет перезапустить запись"""
        if state.phase != Phase.RECORDING:
            return state

        return replace(state, phase=Phase.RESTARTING, error=None)

    @staticmethod
    def handle_ui_toggle_alt_model(state: State, action: UIToggleAltModel) -> State:
        """Пользователь переключил альтернативную ASR модель"""
        return replace(state, llm_enabled=not state.llm_enabled)

    @staticmethod
    def handle_asr_done(state: State, action: ASRDone) -> State:
        """ASR завершено с текстом или ошибкой"""
        # Игнорируем результаты, если мы уже не в фазе PROCESSING
        if state.phase != Phase.PROCESSING:
            return state

        # Обрабатываем ошибку или пустой текст
        if action.error or not action.text:
            return replace(
                state,
                phase=Phase.IDLE,
                error=action.error or "empty asr",
                recognized_text=None,
            )

        # Успех - переходим в IDLE (POST_PROCESSING phase removed)
        return replace(
            state,
            recognized_text=action.text,
            phase=Phase.IDLE,
            error=None,
        )

    @staticmethod
    def handle_llm_done(state: State, action: LLMDone) -> State:
        """LLM пост-обработка завершена"""
        # Игнорируем результаты, если мы уже не в фазе POST_PROCESSING
        if state.phase != Phase.POST_PROCESSING:
            return state

        # Обрабатываем ошибку или пустой текст
        if action.error or not action.text:
            # Можно вернуться к recognized_text, но пока просто ошибка
            return replace(
                state,
                phase=Phase.IDLE,
                error=action.error or "empty llm",
                processed_text=None,
            )

        # Успех
        return replace(state, processed_text=action.text, phase=Phase.IDLE, error=None)

    @staticmethod
    def handle_restart_done(state: State, action: RestartDone) -> State:
        """Перезапуск записи завершён"""
        # Игнорируем результаты, если мы уже не в фазе RESTARTING
        if state.phase != Phase.RESTARTING:
            return state

        if action.success:
            return replace(
                state,
                phase=Phase.RECORDING,
                error=None,
                recognized_text=None,
                processed_text=None,
            )

        return replace(state, phase=Phase.IDLE, error=action.error or "restart failed")

    @staticmethod
    def handle_monitor_changed(state: State, action: MonitorChanged) -> State:
        """Конфигурация монитора изменилась"""
        if action.monitor_name == state.current_monitor_name:
            if action.rel_x is not None and action.rel_y is not None:
                return replace(state, rel_x=action.rel_x, rel_y=action.rel_y)
            return state

        # Если при смене монитора переданы координаты - применяем их сразу
        if action.rel_x is not None and action.rel_y is not None:
            return replace(
                state,
                current_monitor_name=action.monitor_name,
                rel_x=action.rel_x,
                rel_y=action.rel_y,
            )

        return replace(state, current_monitor_name=action.monitor_name)

    @staticmethod
    def handle_window_position_changed(
        state: State, action: WindowPositionChanged
    ) -> State:
        """Позиция окна изменилась"""
        return replace(state, rel_x=action.rel_x, rel_y=action.rel_y)

    @staticmethod
    def reduce(state: State, action: Action) -> State:
        """
        Главный dispatcher reducer'а.

        Перенаправляет действия соответствующим обработчикам и возвращает новое состояние.
        """
        if isinstance(action, UIStart):
            return Reducer.handle_ui_start(state, action)
        elif isinstance(action, UIStop):
            return Reducer.handle_ui_stop(state, action)
        elif isinstance(action, UIRestart):
            return Reducer.handle_ui_restart(state, action)
        elif isinstance(action, UIToggleAltModel):
            return Reducer.handle_ui_toggle_alt_model(state, action)
        elif isinstance(action, ASRDone):
            return Reducer.handle_asr_done(state, action)
        elif isinstance(action, LLMDone):
            return Reducer.handle_llm_done(state, action)
        elif isinstance(action, RestartDone):
            return Reducer.handle_restart_done(state, action)
        elif isinstance(action, MonitorChanged):
            return Reducer.handle_monitor_changed(state, action)
        elif isinstance(action, WindowPositionChanged):
            return Reducer.handle_window_position_changed(state, action)

        return state


class Store:
    """
    Центральное хранилище состояния с dispatch и подпиской.

    Ответственность:
    - Хранить текущее состояние
    - Разрешать изменения состояния через dispatch(action)
    - Уведомлять подписчиков об изменениях состояния
    - Координировать эффекты
    """

    def __init__(
        self, initial: State, reducer: Callable[[State, Action], State], effects: List
    ):
        """
        Инициализация хранилища.

        Args:
            initial: Начальное состояние
            reducer: Чистая функция (State, Action) -> State
            effects: Список обработчиков эффектов
        """
        self._state = initial
        self._reducer = reducer
        self._effects = effects
        self._subs = []
        self._lock = threading.Lock()

    @property
    def state(self) -> State:
        """Получить текущее состояние (потокобезопасное чтение)"""
        with self._lock:
            return self._state

    def subscribe(self, fn: Callable[[State], None]) -> Callable[[], None]:
        """
        Подписаться на изменения состояния.

        Args:
            fn: Callback функция, которая получает новое состояние

        Returns:
            Функция отписки
        """
        self._subs.append(fn)
        # Сразу вызываем с текущим состоянием
        fn(self._state)
        # Возвращаем функцию отписки
        return lambda: self._subs.remove(fn) if fn in self._subs else None

    def dispatch(self, action: Action) -> None:
        """
        Отправить действие для обновления состояния и запуска эффектов.

        Это ЕДИНСТВЕННЫЙ способ изменить состояние. Поток выполнения:
        1. Запустить reducer для получения нового состояния (синхронно, чисто)
        2. Уведомить подписчиков в главном потоке GTK
        3. Запустить эффекты (могут отправить больше действий)

        Args:
            action: Действие для отправки
        """
        # Шаг 1: Запустить reducer (потокобезопасно)
        with self._lock:
            prev = self._state
            next_state = self._reducer(prev, action)
            self._state = next_state

        # Шаг 2: Уведомить подписчиков в главном потоке GTK
        for fn in list(self._subs):
            GLib.idle_add(fn, next_state)

        # Шаг 3: Запустить эффекты (они могут отправить больше действий)
        for eff in self._effects:
            eff.handle(action, prev, next_state, self.dispatch)


# ============================================================================
# REDUX АРХИТЕКТУРА - ЭФФЕКТЫ (ПОБОЧНЫЕ ЭФФЕКТЫ)
# ============================================================================


class StartRecordingEffect:
    """
    Начинает запись когда пользователь нажимает кнопку старт.

    Срабатывает на: UIStart действие при переходе IDLE -> RECORDING
    Побочный эффект: Вызов speech.start()
    При ошибке: Отправляет ASRDone с ошибкой
    """

    def __init__(self, speech):
        """
        Args:
            speech: Реализация SpeechProtocol
        """
        self.speech = speech

    def handle(
        self, action: Action, prev: State, next: State, dispatch: Callable
    ) -> None:
        """Обработка UIStart действия"""
        if (
            isinstance(action, UIStart)
            and prev.phase == Phase.IDLE
            and next.phase == Phase.RECORDING
        ):
            ok = self.speech.start()
            if not ok:
                log("❌ Не удалось запустить запись")
                dispatch(ASRDone(text=None, error="failed to start recording"))


class ASREffect:
    """
    Выполняет распознавание речи когда пользователь останавливает запись.

    Срабатывает на: UIStop действие при переходе RECORDING -> PROCESSING
    Побочный эффект: Запуск speech.stop_and_recognize() async
    Результат: Отправляет ASRDone с текстом или ошибкой
    """

    def __init__(self, speech, async_runner):
        """
        Args:
            speech: Реализация SpeechProtocol
            async_runner: Класс AsyncTaskRunner
        """
        self.speech = speech
        self.async_runner = async_runner

    def handle(
        self, action: Action, prev: State, next: State, dispatch: Callable
    ) -> None:
        """Обработка UIStop действия"""
        if (
            isinstance(action, UIStop)
            and prev.phase == Phase.RECORDING
            and next.phase == Phase.PROCESSING
        ):
            use_alt_model = next.llm_enabled

            def task():
                try:
                    text = self.speech.stop_and_recognize(use_alt_model=use_alt_model)
                    return (text, None)
                except Exception as e:
                    log(f"❌ Ошибка ASR: {e}")
                    return (None, str(e))

            def done(result):
                text, err = result
                dispatch(ASRDone(text=text, error=err))

            self.async_runner.run_async(task, done)


class LLMEffect:
    """
    Выполняет LLM пост-обработку распознанного текста.

    Срабатывает на: ASRDone действие когда llm_enabled=True
    Побочный эффект: Запуск post_processing.process() async
    Результат: Отправляет LLMDone с обработанным текстом или ошибкой
    """

    def __init__(self, post_processing, async_runner):
        """
        Args:
            post_processing: Реализация PostProcessingProtocol
            async_runner: Класс AsyncTaskRunner
        """
        self.pp = post_processing
        self.async_runner = async_runner

    def handle(
        self, action: Action, prev: State, next: State, dispatch: Callable
    ) -> None:
        """Обработка ASRDone действия"""
        # Срабатываем только на ASRDone
        if not isinstance(action, ASRDone):
            return

        # Только если LLM включён и есть текст
        if not next.llm_enabled:
            return
        if not action.text:
            return

        def task():
            try:
                processed = self.pp.process(action.text)
                return (processed, None)
            except Exception as e:
                log(f"❌ Ошибка LLM: {e}")
                return (None, str(e))

        def done(result):
            text, err = result
            dispatch(LLMDone(text=text, error=err))

        self.async_runner.run_async(task, done)


class FinalizeEffect:
    """
    Финализирует обработку текста копированием и вставкой.

    Это КРИТИЧЕСКИЙ эффект, который исправляет баг двойного copy/paste.
    Он обеспечивает ровно ОДНУ финализацию на сессию.

    Срабатывает на:
    1. ASRDone когда llm_enabled=False (копирует распознанный текст)
    2. LLMDone (копирует обработанный текст, или fallback на распознанный)

    Побочные эффекты:
    - Применяет умную обработку текста
    - Копирует в clipboard/primary
    - Автовставка если включена
    """

    def __init__(self, clipboard, paste, glib_module, config):
        """
        Args:
            clipboard: Реализация ClipboardProtocol
            paste: Реализация PasteProtocol
            glib_module: Модуль GLib для timeout_add
            config: AppConfig для задержек
        """
        self.clipboard = clipboard
        self.paste = paste
        self.GLib = glib_module
        self.config = config

    def smart_process(self, state: State, text: str) -> str:
        """Применить умную обработку текста если включена"""
        if not state.smart_text_processing:
            return text

        words = len(text.split())
        if words <= state.smart_short_phrase_words:
            # Короткая фраза: lowercase, убрать точку в конце
            return text.lower().rstrip(".")
        else:
            # Длинная фраза: добавить перевод строки
            return text + " \n"

    def copy_paste(self, state: State, text: str) -> None:
        """Копировать текст и опционально вставить"""
        # Копировать в соответствующий буфер обмена
        if state.copy_method == "clipboard":
            self.clipboard.copy_standard(text)
        else:
            self.clipboard.copy_primary(text)

        # Автовставка если включена
        if state.auto_paste:
            delay_ms = self.config.settings.PASTE_DELAY_MS
            self.GLib.timeout_add(delay_ms, lambda: (self.paste.paste(), False)[1])
            log(f"⌨️  Авто-вставка запланирована ({delay_ms}ms)")

    def handle(
        self, action: Action, prev: State, next: State, dispatch: Callable
    ) -> None:
        """Обработка триггеров финализации"""

        # Случай 1: ASRDone => финализировать распознанным текстом (LLM feature removed)
        if isinstance(action, ASRDone):
            if action.text and next.phase == Phase.IDLE:
                log("✅ Финализация после ASR")
                text = self.smart_process(next, action.text)
                self.copy_paste(next, text)
            return

        # Случай 2: LLMDone => финализировать обработанным текстом (или fallback)
        if isinstance(action, LLMDone):
            if next.phase == Phase.IDLE:
                # Использовать обработанный текст если доступен, иначе fallback на распознанный
                base = action.text or next.recognized_text
                if base:
                    log("✅ Финализация после LLM")
                    text = self.smart_process(next, base)
                    self.copy_paste(next, text)
                else:
                    log("⚠️  Нет текста для финализации")


class RestartEffect:
    """
    Перезапускает запись остановкой, ожиданием и повторным запуском.

    Срабатывает на: UIRestart действие при переходе RECORDING -> RESTARTING
    Побочные эффекты:
    1. Остановка записи (без распознавания)
    2. Ожидание (задержка)
    3. Повторный запуск записи
    Результат: Отправляет RestartDone с успехом/ошибкой
    """

    def __init__(self, speech, async_runner, restart_delay_sec: float):
        """
        Args:
            speech: Реализация SpeechProtocol
            async_runner: Класс AsyncTaskRunner
            restart_delay_sec: Задержка между остановкой и запуском
        """
        self.speech = speech
        self.async_runner = async_runner
        self.delay = restart_delay_sec

    def handle(
        self, action: Action, prev: State, next: State, dispatch: Callable
    ) -> None:
        """Обработка UIRestart действия"""
        if (
            isinstance(action, UIRestart)
            and prev.phase == Phase.RECORDING
            and next.phase == Phase.RESTARTING
        ):
            log("🔄 Перезапуск записи...")

            def task():
                try:
                    # Остановка без распознавания
                    self.speech.stop()
                    log(f"⏸️  Запись остановлена, ожидание {self.delay}s...")

                    # Задержка
                    time.sleep(self.delay)

                    # Запуск снова
                    ok = self.speech.start()
                    return (ok, None)
                except Exception as e:
                    log(f"❌ Ошибка перезапуска: {e}")
                    return (False, str(e))

            def done(result):
                ok, err = result
                dispatch(RestartDone(success=ok, error=err))

            self.async_runner.run_async(task, done)


class SettingsPersistenceEffect:
    """
    Сохраняет настройки пользователя в JSON файл при их изменении.

    Срабатывает на: UIToggleAltModel и другие actions, меняющие настройки
    Побочный эффект: Запись settings.json в ~/.config/float-speech-to-text/
    """

    def __init__(self, settings_file: str):
        """
        Args:
            settings_file: Путь к файлу настроек (JSON)
        """
        self.settings_file = settings_file
        # Создаём директорию если не существует
        os.makedirs(os.path.dirname(settings_file), exist_ok=True)

    def handle(
        self, action: Action, prev: State, next: State, dispatch: Callable
    ) -> None:
        """Обработка изменений настроек"""
        # Проверяем, изменились ли настройки
        settings_changed = (
            prev.llm_enabled != next.llm_enabled
            or prev.auto_paste != next.auto_paste
            or prev.copy_method != next.copy_method
            or prev.smart_text_processing != next.smart_text_processing
            or prev.smart_short_phrase_words != next.smart_short_phrase_words
        )

        if settings_changed:
            self._save_settings(next)

    def _save_settings(self, state: State) -> None:
        """Сохраняет настройки в JSON файл"""
        try:
            settings = {
                "llm_enabled": state.llm_enabled,
                "auto_paste": state.auto_paste,
                "copy_method": state.copy_method,
                "smart_text_processing": state.smart_text_processing,
                "smart_short_phrase_words": state.smart_short_phrase_words,
            }

            with open(self.settings_file, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)

            log(f"💾 Настройки сохранены в {self.settings_file}")
        except Exception as e:
            log(f"❌ Ошибка сохранения настроек: {e}")

    @staticmethod
    def load_settings(settings_file: str) -> dict:
        """Загружает настройки из JSON файла"""
        try:
            if os.path.exists(settings_file):
                with open(settings_file, "r", encoding="utf-8") as f:
                    settings = json.load(f)
                log(f"📂 Настройки загружены из {settings_file}")
                return settings
        except Exception as e:
            log(f"❌ Ошибка загрузки настроек: {e}")

        return {}


class WindowPersistenceEffect:
    """
    Управляет загрузкой и сохранением позиции окна через Redux.

    Срабатывает на:
    1. MonitorChanged: Загружает позицию для нового монитора и отправляет WindowPositionChanged
    2. WindowPositionChanged: Сохраняет новую позицию для текущего монитора
    """

    def __init__(self, monitor_manager, window_persistence, config):
        """
        Args:
            monitor_manager: MonitorManager для расчётов позиций
            window_persistence: WindowPositionPersistence для I/O
            config: AppConfig
        """
        self.mm = monitor_manager
        self.wp = window_persistence
        self.config = config

    def handle(
        self, action: Action, prev: State, next: State, dispatch: Callable
    ) -> None:
        """Обработка событий монитора и позиции"""

        # Сохраняем позицию только если она была изменена вручную (перетаскивание)
        if isinstance(action, WindowPositionChanged) and action.is_manual:
            if next.current_monitor_name:
                self.wp.save_position(
                    next.current_monitor_name, action.rel_x, action.rel_y
                )
                self.wp.save_last_monitor(next.current_monitor_name)


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================


def log(message):
    """Вывод отладочной информации в stderr"""
    print(message, file=sys.stderr)


def load_prompt_from_file(file_path: str, default_prompt: str) -> str:
    """Загружает текст промпта из файла"""
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        else:
            log(f"⚠️  Файл с промптом не найден: {file_path}")
    except Exception as e:
        log(f"❌ Ошибка загрузки промпта из файла: {e}")
    return default_prompt


class MonitorManager:
    """Управление состоянием мониторов и относительным позиционированием окна"""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config
        self.display = None
        self.monitors_available = True
        self.on_stable_change = None

        # Состояние ретраев для "не готовых" мониторов
        self.retry_id = None
        self.retry_count = 0

    def get_monitor_at_cursor(self) -> Optional[Gdk.Monitor]:
        """Возвращает монитор, на котором находится курсор мыши, или первый доступный"""
        if not self.display:
            self.display = Gdk.Display.get_default()

        if not self.display:
            log("⚠️  Не удалось получить display")
            return None

        try:
            # Получаем устройство указателя
            seat = self.display.get_default_seat()
            if not seat:
                log("⚠️  Не удалось получить seat, используем первый монитор")
                return self.get_first_monitor()

            pointer = seat.get_pointer()
            if not pointer:
                log("⚠️  Не удалось получить pointer, используем первый монитор")
                return self.get_first_monitor()

            # Получаем позицию курсора
            screen, x, y = pointer.get_position()

            # Находим монитор по координатам курсора
            monitor = self.display.get_monitor_at_point(int(x), int(y))

            if monitor:
                log(f"📺 Монитор с курсором: {self.get_monitor_identifier(monitor)}")
                return monitor
            else:
                # Курсор не на мониторе (может быть между мониторами или монитор только включился)
                log("⚠️  Курсор не на мониторе, используем первый доступный монитор")
                return self.get_first_monitor()
        except Exception as e:
            log(
                f"⚠️  Ошибка определения монитора по курсору: {e}, используем первый монитор"
            )
            return self.get_first_monitor()

    def get_first_monitor(self) -> Optional[Gdk.Monitor]:
        """Возвращает первый доступный монитор"""
        if not self.display:
            self.display = Gdk.Display.get_default()

        if not self.display:
            log("⚠️  Display не доступен в get_first_monitor")
            return None

        n_monitors = self.display.get_n_monitors()
        log(f"🔍 get_first_monitor: найдено {n_monitors} мониторов")

        if n_monitors > 0:
            monitor = self.display.get_monitor(0)
            if monitor:
                model = self.get_monitor_identifier(monitor)
                log(f"📺 Используется первый монитор: {model}")
                return monitor
            else:
                log("⚠️  Монитор с индексом 0 вернул None")
        else:
            log("⚠️  Нет доступных мониторов")

        return None

    def get_monitor_identifier(self, monitor: Gdk.Monitor) -> Optional[str]:
        """
        Получает идентификатор монитора.
        Возвращает None, если монитор еще не инициализирован (0x0 geometry и нет модели).
        """
        if not monitor:
            return None

        # 1. Проверяем геометрию - если 0x0, то монитор скорее всего не готов
        try:
            geom = monitor.get_geometry()
            if geom.width <= 1 or geom.height <= 1:
                # Если размеры почти нулевые, проверяем модель. Если и её нет - монитор не готов.
                if not hasattr(monitor, "get_model") or not monitor.get_model():
                    log(
                        f"⚠️  Монитор имеет нулевую геометрию и нет модели - он не готов"
                    )
                    return None
        except Exception:
            pass

        # 2. Пробуем get_model()
        try:
            if hasattr(monitor, "get_model"):
                model = monitor.get_model()
                if model:
                    return model
        except Exception:
            pass

        # 3. Пробуем manufacturer + connector
        try:
            manufacturer = None
            if hasattr(monitor, "get_manufacturer"):
                manufacturer = monitor.get_manufacturer()

            connector = None
            if hasattr(monitor, "get_connector"):
                connector = monitor.get_connector()

            if manufacturer and connector:
                return f"{manufacturer}_{connector}"
            elif connector:
                return connector
            elif manufacturer:
                return manufacturer
        except Exception:
            pass

        # 4. Fallback на геометрию
        try:
            geom = monitor.get_geometry()
            # Если дошли сюда с 0x0, все методы выше не сработали.
            if geom.width <= 1 or geom.height <= 1:
                return None
            return f"Monitor_{geom.width}x{geom.height}_{geom.x}x{geom.y}"
        except Exception as e:
            log(f"⚠️  Ошибка получения идентификатора монитора: {e}")
            return None

    def get_monitor_by_name(self, monitor_name: str) -> Optional[Gdk.Monitor]:
        """Находит монитор по его имени/модели"""
        if not self.display:
            self.display = Gdk.Display.get_default()

        if not self.display:
            return None

        n_monitors = self.display.get_n_monitors()
        for i in range(n_monitors):
            monitor = self.display.get_monitor(i)
            if monitor and self.get_monitor_identifier(monitor) == monitor_name:
                return monitor

        return None

    def get_monitor_geometry(self, monitor: Gdk.Monitor) -> dict:
        """Возвращает геометрию монитора (ширина, высота, позиция)"""
        geometry = monitor.get_geometry()
        return {
            "x": geometry.x,
            "y": geometry.y,
            "width": geometry.width,
            "height": geometry.height,
        }

    def calculate_relative_position(
        self,
        margin_right: int,
        margin_top: int,
        window_width: int,
        window_height: int,
        monitor: Gdk.Monitor,
    ) -> tuple[float, float]:
        """
        Вычисляет относительную позицию центра окна (в процентах) от размера монитора

        Args:
            margin_right, margin_top: Абсолютные отступы от правого/верхнего края (GtkLayerShell margins)
            window_width, window_height: Размеры окна в пикселях
            monitor: Монитор для расчета

        Returns:
            (rel_center_x, rel_center_y): Относительные координаты центра окна от 0.0 до 1.0
        """
        geometry = self.get_monitor_geometry(monitor)
        monitor_width = geometry["width"]
        monitor_height = geometry["height"]

        # Вычисляем абсолютную позицию центра окна
        # margin_right - это расстояние от правого края монитора до правого края окна
        # Следовательно, правый край окна находится на (monitor_width - margin_right)
        # А центр окна на (monitor_width - margin_right - window_width / 2)
        center_x_abs = monitor_width - margin_right - window_width / 2.0

        # margin_top - это расстояние от верхнего края монитора до верхнего края окна
        # Центр окна на (margin_top + window_height / 2)
        center_y_abs = margin_top + window_height / 2.0

        # Преобразуем в относительные координаты (0.0 - 1.0)
        rel_center_x = center_x_abs / monitor_width if monitor_width > 0 else 0.5
        rel_center_y = center_y_abs / monitor_height if monitor_height > 0 else 0.5

        # Ограничиваем значения диапазоном [0.0, 1.0]
        rel_center_x = max(0.0, min(1.0, rel_center_x))
        rel_center_y = max(0.0, min(1.0, rel_center_y))

        log(
            f"🧮 Относительная позиция центра: ({rel_center_x:.3f}, {rel_center_y:.3f})"
        )

        return (rel_center_x, rel_center_y)

    def calculate_absolute_position(
        self,
        rel_center_x: float,
        rel_center_y: float,
        window_width: int,
        window_height: int,
        monitor: Gdk.Monitor,
    ) -> tuple[int, int]:
        """
        Вычисляет абсолютную позицию (margins) из относительной позиции центра окна

        Args:
            rel_center_x, rel_center_y: Относительные координаты центра окна (0.0-1.0)
            window_width, window_height: Размеры окна в пикселях
            monitor: Монитор для расчета

        Returns:
            (margin_right, margin_top): Абсолютные отступы для GtkLayerShell (TOP + RIGHT anchors)
        """
        geometry = self.get_monitor_geometry(monitor)
        monitor_width = geometry["width"]
        monitor_height = geometry["height"]

        # Вычисляем абсолютную позицию центра окна
        center_x_abs = rel_center_x * monitor_width
        center_y_abs = rel_center_y * monitor_height

        # Вычисляем позицию правого верхнего угла окна
        # Правый край окна: center_x + window_width / 2
        # margin_right = monitor_width - (center_x + window_width / 2)
        margin_right = monitor_width - (center_x_abs + window_width / 2.0)

        # Верхний край окна: center_y - window_height / 2
        # margin_top = center_y - window_height / 2
        margin_top = center_y_abs - window_height / 2.0

        # Ограничиваем значения, чтобы окно не выходило за пределы монитора
        margin_right = max(0, min(monitor_width - window_width, margin_right))
        margin_top = max(0, min(monitor_height - window_height, margin_top))

        return (int(margin_right), int(margin_top))

    def start_monitoring(self, display: Gdk.Display, on_stable_change: Callable):
        """
        Запускает мониторинг изменений дисплеев.
        Колбэк on_stable_change будет вызван только когда монитор определен и готов.
        """
        self.display = display
        self.on_stable_change = on_stable_change

        # Подписываемся на изменения конфигурации мониторов
        display.connect("monitor-added", self._handle_monitor_event)
        display.connect("monitor-removed", self._handle_monitor_event)

        log("👀 Мониторинг дисплеев запущен")

    def _handle_monitor_event(self, display, monitor=None):
        """Внутренний обработчик событий монитора с логикой ретраев"""
        log("📺 Обнаружено изменение конфигурации мониторов (MonitorManager)")

        # Отменяем текущий ретрай если есть
        if self.retry_id:
            GLib.source_remove(self.retry_id)
            self.retry_id = None

        if not self.check_monitors_available():
            log("⚠️  Все мониторы отключены")
            if self.on_stable_change:
                self.on_stable_change(None)
            self.retry_count = 0
            return

        # Пытаемся найти активный монитор
        active_monitor = self.find_active_monitor()

        if active_monitor:
            monitor_name = self.get_monitor_identifier(active_monitor)
            if monitor_name:
                log(f"✅ Монитор готов: {monitor_name}")
                self.retry_count = 0
                if self.on_stable_change:
                    self.on_stable_change(active_monitor)
                return

        # Если мониторы есть, но не готовы - планируем ретрай
        self._schedule_retry(display, monitor)

    def _schedule_retry(self, display, monitor):
        """Планирует повторную проверку мониторов"""
        if self.retry_count < 15:  # 15 * 200ms = 3 сек
            self.retry_count += 1
            log(
                f"⏳ Мониторы не готовы. Повтор через 200мс (попытка {self.retry_count})"
            )
            self.retry_id = GLib.timeout_add(
                200, lambda: (self._handle_monitor_event(display, monitor), False)[1]
            )
        else:
            log("⚠️  Не удалось найти готовый монитор после всех попыток")
            self.retry_count = 0
            # Если так и не нашли ничего за 3 секунды, уведомляем о потере
            if self.on_stable_change:
                self.on_stable_change(None)

    def find_active_monitor(self) -> Optional[Gdk.Monitor]:
        """
        Стратегия поиска лучшего монитора для размещения окна:
        1. Пробуем монитор с курсором
        2. Пробуем последний использованный монитор (если config доступен)
        3. Берем первый попавшийся
        """
        try:
            # 1. По курсору
            monitor = self.get_monitor_at_cursor()
            if monitor:
                name = self.get_monitor_identifier(monitor)
                if name:
                    return monitor

            # 2. По конфигу (последний известный)
            if self.config:
                last_name = self.config.window.get_last_monitor()
                if last_name:
                    monitor = self.get_monitor_by_name(last_name)
                    if monitor and self.get_monitor_identifier(monitor):
                        return monitor

            # 3. Fallback: Первый попавшийся
            return self.get_first_monitor()
        except Exception as e:
            log(f"❌ Ошибка при поиске монитора: {e}")
            return None

    def check_monitors_available(self) -> bool:
        """Проверяет наличие активных мониторов"""
        if not self.display:
            self.display = Gdk.Display.get_default()

        if not self.display:
            self.monitors_available = False
            return False

        n_monitors = self.display.get_n_monitors()
        self.monitors_available = n_monitors > 0

        log(f"📺 Доступно мониторов: {n_monitors}")

        return self.monitors_available


# ============================================================================
# ПРОТОКОЛЫ (АБСТРАКЦИИ)
# ============================================================================


class ClipboardProtocol(Protocol):
    """Протокол для сервиса работы с буфером обмена"""

    def copy_standard(self, text: str) -> bool:
        """Копирует текст в стандартный буфер обмена (Ctrl+V)"""
        ...

    def copy_primary(self, text: str) -> bool:
        """Копирует текст в primary selection (средняя кнопка мыши)"""
        ...


class PasteProtocol(Protocol):
    """Протокол для сервиса вставки текста"""

    def paste(self) -> bool:
        """Эмулирует вставку текста в зависимости от настройки"""
        ...


class SpeechProtocol(Protocol):
    """Протокол для сервиса записи и распознавания речи"""

    @property
    def is_recording(self) -> bool:
        """Возвращает True, если идёт запись"""
        ...

    def start(self) -> bool:
        """Начинает запись аудио"""
        ...

    def stop(self) -> None:
        """Останавливает запись БЕЗ распознавания"""
        ...

    def stop_and_recognize(self) -> Optional[str]:
        """Останавливает запись и распознаёт речь"""
        ...


class PostProcessingProtocol(Protocol):
    """Протокол для сервиса пост-обработки текста"""

    def process(self, text: str) -> str:
        """Обрабатывает текст с помощью LLM"""
        ...


# ============================================================================
# КОНФИГУРАЦИЯ И КОНСТАНТЫ
# ============================================================================


def get_env_bool(name: str, default: bool) -> bool:
    """Получает булево значение из переменной окружения"""
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes", "on")


def get_env_int(name: str, default: int) -> int:
    """Получает целое число из переменной окружения"""
    try:
        return int(os.environ.get(name, default))
    except (ValueError, TypeError):
        return default


def get_env_float(name: str, default: float) -> float:
    """Получает число с плавающей точкой из переменной окружения"""
    try:
        return float(os.environ.get(name, default))
    except (ValueError, TypeError):
        return default


class AudioConfig:
    """Настройки для аудио записи и распознавания"""

    SAMPLE_RATE = get_env_int("FSTT_WAV_SAMPLE_RATE", 16000)
    CHANNELS = get_env_int("FSTT_WAV_CHANNELS", 1)
    DTYPE = "int16"
    SAMPLE_WIDTH = 2
    MODEL_NAME = os.environ.get("FSTT_ONNX_ASR_MODEL", "gigaam-v3-e2e-rnnt")
    ALT_ASR_MODEL = os.environ.get(
        "FSTT_ALT_ASR_MODEL", "onnx-community/whisper-medium.en_timestamped"
    )
    WAV_FILE = "recording.wav"


class UIConfig:
    """Настройки для пользовательского интерфейса"""

    DEFAULT_WINDOW_X = 20
    DEFAULT_WINDOW_Y = 20
    ICON_RECORD = "●"
    ICON_STOP = "■"
    ICON_PROCESSING = "⋯"
    ICON_CLOSE = "✕"
    ICON_RESTART = "↻"
    ICON_ALT_MODEL_ON = "EN"  # Whisper (English model)
    ICON_ALT_MODEL_OFF = "RU"  # Gigaam (Russian model)
    BOX_SPACING = 5
    BOX_MARGIN = 10
    MOUSE_BUTTON_LEFT = 1

    CSS_STYLES = b"""
window {
    background-color: rgba(0, 0, 0, 0.1);
    border-radius: 10px;
}
button {
    background-color: rgba(0, 0, 0, 0.3);
    color: rgba(255, 255, 255, 0.5);
    border-radius: 5px;
    border: none;
    font-size: 20px;
    padding: 5px 0px;
    min-width: 35px;
}
button:hover {
    background-color: rgba(60, 60, 60, 0.3);
}
button:disabled {
    background-color: rgba(0, 0, 0, 0.3);
    color: rgba(120, 120, 120, 0.5);
}
.record-button label {
    margin-top: -2px;
    margin-bottom: 2px;
}
.restart-button label {
    margin-top: 1px;
    margin-bottom: -1px;
}
.close-button label {
    margin-top: 0px;
    margin-bottom: 0px;
}
.autopaste-button label {
    margin-top: 0px;
    margin-bottom: 0px;
}
"""


class AppSettings:
    """Настройки поведения приложения"""

    APP_ID = "com.example.voice_recognition"
    COPY_METHOD = os.environ.get(
        "FSTT_CLIPBOARD_COPY_METHOD", "clipboard"
    )  # Варианты: "primary", "clipboard"
    AUTO_PASTE = get_env_bool("FSTT_CLIPBOARD_PASTE_ENABLED", True)
    LLM_ENABLED = get_env_bool("FSTT_LLM_ENABLED", True)
    LLM_PROMPT_FILE = os.environ.get("FSTT_LLM_PROMPT_FILE", "prompt.md")
    LLM_TEMPERATURE = get_env_float("FSTT_LLM_TEMPERATURE", 1.0)
    LLM_MAX_RETRIES = get_env_int("FSTT_LLM_MAX_RETRIES", 2)
    LLM_TIMEOUT_SEC = get_env_int("FSTT_LLM_TIMEOUT_SEC", 60)
    SMART_TEXT_PROCESSING = get_env_bool(
        "FSTT_POSTPROCESSING_ENABLED", False
    )  # Включает умную обработку текста (короткие/длинные фразы)
    SMART_TEXT_SHORT_PHRASE = get_env_int(
        "FSTT_POSTPROCESSING_WORD_THRESHOLD", 3
    )  # Максимальное количество слов для постобработки обработки коротких фраз

    # Настройки OpenAI из переменных окружения
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    # Таймауты и задержки
    PASTE_DELAY_MS = get_env_int("FSTT_CLIPBOARD_PASTE_DELAY_MS", 200)
    RESTART_DELAY_SEC = get_env_float("FSTT_RECORD_RESTART_DELAY_SEC", 0.1)


class WindowPositionPersistence:
    """Управление сохранением и загрузкой позиции окна для каждого монитора отдельно"""

    CONFIG_FILE = os.path.expanduser("~/.config/voice-recognition-window.json")
    DEFAULT_CENTER_X = 0.5  # Центр по горизонтали
    DEFAULT_CENTER_Y = 0.1  # 10% от верха

    @classmethod
    def load_position(cls, monitor_name: str) -> tuple[float, float]:
        """
        Загружает сохранённую относительную позицию центра окна для указанного монитора

        Args:
            monitor_name: Имя/модель монитора

        Returns:
            (center_x, center_y): Относительные координаты центра окна (0.0-1.0)
        """
        try:
            if os.path.exists(cls.CONFIG_FILE):
                with open(cls.CONFIG_FILE, "r") as f:
                    config = json.load(f)
                    monitors = config.get("monitors", {})
                    monitor_config = monitors.get(monitor_name, {})

                    center_x = monitor_config.get("center_x", cls.DEFAULT_CENTER_X)
                    center_y = monitor_config.get("center_y", cls.DEFAULT_CENTER_Y)

                    log(
                        f"📂 Загружена позиция для монитора {monitor_name}: center=({center_x:.3f}, {center_y:.3f})"
                    )
                    return (center_x, center_y)
        except Exception as e:
            log(f"⚠️  Ошибка загрузки конфига: {e}")

        log(f"📂 Используется дефолтная позиция для монитора {monitor_name}")
        return (cls.DEFAULT_CENTER_X, cls.DEFAULT_CENTER_Y)

    @classmethod
    def save_position(cls, monitor_name: str, center_x: float, center_y: float) -> None:
        """
        Сохраняет относительную позицию центра окна для указанного монитора

        Args:
            monitor_name: Имя/модель монитора
            center_x, center_y: Относительные координаты центра окна (0.0-1.0)
        """
        try:
            config_dir = os.path.dirname(cls.CONFIG_FILE)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir)

            # Загружаем существующий конфиг
            config = {}
            if os.path.exists(cls.CONFIG_FILE):
                with open(cls.CONFIG_FILE, "r") as f:
                    config = json.load(f)

            # Обновляем позицию для монитора
            if "monitors" not in config:
                config["monitors"] = {}

            config["monitors"][monitor_name] = {
                "center_x": center_x,
                "center_y": center_y,
            }

            # Сохраняем
            with open(cls.CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=2)

            log(
                f"💾 Сохранена позиция для монитора {monitor_name}: center=({center_x:.3f}, {center_y:.3f})"
            )
        except Exception as e:
            log(f"⚠️  Ошибка сохранения конфига: {e}")

    @classmethod
    def get_last_monitor(cls) -> Optional[str]:
        """Возвращает имя последнего активного монитора"""
        try:
            if os.path.exists(cls.CONFIG_FILE):
                with open(cls.CONFIG_FILE, "r") as f:
                    config = json.load(f)
                    last_monitor = config.get("last_monitor")
                    if last_monitor:
                        log(f"📂 Последний монитор: {last_monitor}")
                    return last_monitor
        except Exception as e:
            log(f"⚠️  Ошибка загрузки конфига: {e}")

        return None

    @classmethod
    def save_last_monitor(cls, monitor_name: str) -> None:
        """Сохраняет имя последнего активного монитора"""
        try:
            config_dir = os.path.dirname(cls.CONFIG_FILE)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir)

            # Загружаем существующий конфиг
            config = {}
            if os.path.exists(cls.CONFIG_FILE):
                with open(cls.CONFIG_FILE, "r") as f:
                    config = json.load(f)

            config["last_monitor"] = monitor_name

            # Сохраняем
            with open(cls.CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=2)

            log(f"💾 Сохранён последний монитор: {monitor_name}")
        except Exception as e:
            log(f"⚠️  Ошибка сохранения конфига: {e}")

    @classmethod
    def load(cls) -> tuple[int, int]:
        """
        Устаревший метод для обратной совместимости
        Возвращает дефолтные значения
        """
        return (int(UIConfig.DEFAULT_WINDOW_X), int(UIConfig.DEFAULT_WINDOW_Y))


class AppConfig:
    """
    Объединённая конфигурация приложения.

    Предоставляет единую точку доступа ко всем настройкам через ссылки на под-конфиги.

    Использование:
        config = AppConfig()
        config.audio.SAMPLE_RATE  # Аудио настройки
        config.ui.ICON_RECORD     # UI настройки
        config.settings.AUTO_PASTE # Настройки поведения
        config.window.load()      # Работа с позицией окна
    """

    # Ссылки на под-конфиги
    audio = AudioConfig
    ui = UIConfig
    settings = AppSettings
    window = WindowPositionPersistence


# ============================================================================
# ФАБРИКА СЕРВИСОВ
# ============================================================================


class ServiceFactory:
    """
    Фабрика для создания сервисов с их зависимостями.

    Поддерживает Dependency Injection через конструктор для легкой замены реализаций.
    """

    def __init__(
        self,
        clipboard_class: type = None,
        paste_class: type = None,
        speech_class: type = None,
        post_processing_class: type = None,
    ):
        """
        Инициализирует фабрику с возможностью внедрения зависимостей.

        Args:
            clipboard_class: Класс для создания сервиса буфера обмена (по умолчанию ClipboardService)
            paste_class: Класс для создания сервиса вставки (по умолчанию PasteService)
            speech_class: Класс для создания сервиса распознавания речи (по умолчанию SpeechService)
            post_processing_class: Класс для создания сервиса пост-обработки (по умолчанию PostProcessingService)
        """
        # Используем отложенную инициализацию дефолтных классов, чтобы избежать circular dependencies
        self._clipboard_class = clipboard_class
        self._paste_class = paste_class
        self._speech_class = speech_class
        self._post_processing_class = post_processing_class

    @property
    def clipboard_class(self):
        """Возвращает класс сервиса буфера обмена (ленивая инициализация)"""
        if self._clipboard_class is None:
            return ClipboardService
        return self._clipboard_class

    @property
    def paste_class(self):
        """Возвращает класс сервиса вставки (ленивая инициализация)"""
        if self._paste_class is None:
            return PasteService
        return self._paste_class

    @property
    def speech_class(self):
        """Возвращает класс сервиса распознавания речи (ленивая инициализация)"""
        if self._speech_class is None:
            return SpeechService
        return self._speech_class

    @property
    def post_processing_class(self):
        """Возвращает класс сервиса пост-обработки (ленивая инициализация)"""
        if self._post_processing_class is None:
            return PostProcessingService
        return self._post_processing_class

    def create_clipboard(self) -> ClipboardProtocol:
        """Создаёт сервис буфера обмена"""
        return self.clipboard_class()

    def create_paste(self, copy_method: str) -> PasteProtocol:
        """Создаёт сервис вставки текста"""
        return self.paste_class(copy_method)

    def create_speech(self, config: "AppConfig") -> SpeechProtocol:
        """Создаёт сервис распознавания речи"""
        return self.speech_class(config)

    def create_post_processing(self, config: "AppConfig") -> PostProcessingProtocol:
        """Создаёт сервис пост-обработки"""
        return self.post_processing_class(config)

    def create_all_services(
        self, config: "AppConfig"
    ) -> tuple[
        SpeechProtocol, ClipboardProtocol, PasteProtocol, PostProcessingProtocol
    ]:
        """Создаёт все необходимые сервисы"""
        speech = self.create_speech(config)
        clipboard = self.create_clipboard()
        paste = self.create_paste(config.settings.COPY_METHOD)
        post_processing = self.create_post_processing(config)
        return speech, clipboard, paste, post_processing


# ============================================================================
# СЕРВИСЫ
# ============================================================================


class ClipboardService:
    """Сервис для работы с буфером обмена (clipboard и primary selection)"""

    def copy_standard(self, text):
        """Копирует текст в стандартный буфер обмена (Ctrl+V)"""
        try:
            import pyclip

            pyclip.copy(text)
            log("📋 Скопировано в буфер обмена")
            return True
        except ImportError:
            log("⚠️  pyclip не установлен, используйте: pip install pyclip")
            log(
                "⚠️  Или установите wl-clipboard для Wayland: sudo pacman -S wl-clipboard"
            )
            return False
        except Exception as e:
            log(f"❌ Ошибка копирования в буфер обмена: {e}")
            return False

    def copy_primary(self, text):
        """Копирует текст в primary selection (средняя кнопка мыши)"""
        # Пробуем wl-copy для Wayland
        if shutil.which("wl-copy"):
            return self._copy_primary_wl(text)

        # Пробуем xsel для X11
        if shutil.which("xsel"):
            return self._copy_primary_xsel(text)

        # Пробуем xclip для X11
        if shutil.which("xclip"):
            return self._copy_primary_xclip(text)

        # Резервный вариант: GTK Clipboard API
        log("⚠️  Системные команды не найдены, пробую GTK Clipboard API...")
        log("💡 Установите wl-clipboard для Wayland: sudo pacman -S wl-clipboard")
        log("💡 Или установите xsel для X11: sudo pacman -S xsel")
        return self._copy_primary_gtk(text)

    def _copy_primary_wl(self, text):
        """Копирует через wl-copy (Wayland)"""
        try:
            escaped_text = shlex.quote(text)
            subprocess.Popen(
                f"printf %s {escaped_text} | wl-copy --primary &",
                shell=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            log("🖱️  Скопировано в primary selection через wl-copy")
            return True
        except Exception as e:
            log(f"❌ Ошибка при использовании wl-copy: {e}")
            return False

    def _copy_primary_xsel(self, text):
        """Копирует через xsel (X11)"""
        try:
            process = subprocess.Popen(
                ["xsel", "--primary", "--input"],
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = process.communicate(input=text.encode("utf-8"))

            if process.returncode == 0:
                log("🖱️  Скопировано в primary selection через xsel")
                return True
            else:
                log(
                    f"⚠️  xsel вернул код {process.returncode}: {stderr.decode('utf-8', errors='ignore')}"
                )
                return False
        except Exception as e:
            log(f"❌ Ошибка при использовании xsel: {e}")
            return False

    def _copy_primary_xclip(self, text):
        """Копирует через xclip (X11)"""
        try:
            process = subprocess.Popen(
                ["xclip", "-selection", "primary"],
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = process.communicate(input=text.encode("utf-8"))

            if process.returncode == 0:
                log("🖱️  Скопировано в primary selection через xclip")
                return True
            else:
                log(
                    f"⚠️  xclip вернул код {process.returncode}: {stderr.decode('utf-8', errors='ignore')}"
                )
                return False
        except Exception as e:
            log(f"❌ Ошибка при использовании xclip: {e}")
            return False

    def _copy_primary_gtk(self, text):
        """Копирует через GTK Clipboard API"""
        try:
            clipboard = Gtk.Clipboard.get(Gdk.SELECTION_PRIMARY)
            clipboard.set_text(text, -1)
            clipboard.store()
            log("🖱️  Скопировано в primary selection через GTK")
            return True
        except Exception as e:
            log(f"❌ Ошибка копирования в primary selection через GTK: {e}")
            return False


class PasteService:
    """Сервис для вставки текста через эмуляцию клавиатуры (wtype)"""

    def __init__(self, copy_method: str):
        """
        Инициализация сервиса вставки

        Args:
            copy_method: Метод копирования ("clipboard", "primary")
        """
        self.copy_method = copy_method

    def paste(self):
        """Эмулирует вставку текста в зависимости от настройки copy_method"""
        if self.copy_method == "primary":
            return self._paste_primary()
        elif self.copy_method == "clipboard":
            return self._paste_clipboard()
        else:
            log(f"⚠️  Неизвестный метод копирования: {self.copy_method}")
            return self._paste_clipboard()

    def _paste_clipboard(self):
        """Эмулирует нажатие Ctrl+V для вставки из стандартного буфера обмена"""
        if not shutil.which("wtype"):
            log("⚠️  wtype не найден. Установите wtype: sudo pacman -S wtype")
            return False

        try:
            # wtype -M ctrl -k v -m ctrl
            subprocess.run(["wtype", "-M", "ctrl", "-k", "v", "-m", "ctrl"], check=True)
            log("⌨️  Выполнена вставка из clipboard (Ctrl+V) через wtype")
            return True
        except Exception as e:
            log(f"❌ Ошибка при выполнении wtype: {e}")
            return False

    def _paste_primary(self):
        """Эмулирует нажатие Shift+Insert для вставки из primary selection"""
        if not shutil.which("wtype"):
            log("⚠️  wtype не найден. Установите wtype: sudo pacman -S wtype")
            return False

        try:
            # wtype -M shift -k Insert -m shift
            subprocess.run(
                ["wtype", "-M", "shift", "-k", "Insert", "-m", "shift"], check=True
            )
            log("⌨️  Выполнена вставка из primary selection (Shift+Insert) через wtype")
            return True
        except Exception as e:
            log(f"❌ Ошибка при выполнении wtype: {e}")
            return False


class SpeechService:
    """Сервис для записи и распознавания речи"""

    def __init__(self, config):
        self.config = config
        self.recording = []
        self.is_recording = False
        self.stream = None
        self.model = None
        self.whisper_model = None
        self.whisper_pipeline = None
        self._stream_lock = threading.Lock()

        # Запускаем поток сразу при инициализации
        self._init_stream()

    def _init_stream(self):
        """Инициализирует и запускает постоянно работающий поток"""

        def callback(indata, _frames, time, status):
            if status:
                log(f"⚠️  Статус: {status}")

            # Пишем ВСЕГДА, независимо от состояния
            # Но добавляем только если запись активна
            if self.is_recording:
                with self._stream_lock:
                    self.recording.append(indata.copy())

        # Создаём и запускаем поток
        self.stream = sd.InputStream(
            samplerate=self.config.audio.SAMPLE_RATE,
            channels=self.config.audio.CHANNELS,
            dtype=self.config.audio.DTYPE,
            callback=callback,
        )
        self.stream.start()
        log("🎤 Аудио-поток инициализирован и прогрет")

    def start(self):
        """Начинает запись аудио"""
        if self.is_recording:
            return False

        log("🎤 Начинаю запись...")

        # Атомарная операция: сначала останавливаем запись, очищаем буфер, затем запускаем
        # Это гарантирует, что callback не добавит старые данные в новый буфер
        with self._stream_lock:
            # Сначала сбрасываем флаг (на всякий случай)
            self.is_recording = False
            # Очищаем буфер от любых остатков
            self.recording = []
            # Только теперь включаем запись - буфер чист
            self.is_recording = True

        log("✅ Запись началась (поток уже был готов)")
        return True

    def stop(self):
        """Останавливает запись БЕЗ распознавания (для перезапуска)"""
        if not self.is_recording:
            return

        log("⏹️  Запись остановлена (без распознавания)")

        # НЕ закрываем поток! Он работает постоянно
        # Атомарно останавливаем запись и очищаем буфер
        with self._stream_lock:
            self.is_recording = False
            self.recording = []

    def stop_and_recognize(self, use_alt_model=False):
        """Останавливает запись и распознаёт речь"""
        if not self.is_recording:
            return None

        log("⏹️  Запись остановлена")

        # НЕ закрываем поток! Он работает постоянно
        # Атомарно останавливаем запись и копируем буфер
        with self._stream_lock:
            self.is_recording = False
            recording_copy = self.recording.copy()

        if not recording_copy:
            log("❌ Ничего не записано")
            return None

        # Объединяем все буферы
        audio_data = np.concatenate(recording_copy, axis=0)
        duration = len(audio_data) / self.config.audio.SAMPLE_RATE
        log(f"✅ Записано {len(audio_data)} сэмплов ({duration:.2f} сек)")

        # Сохраняем в WAV файл
        self._save_wav(audio_data)

        # Распознаём речь
        return self._recognize(use_alt_model=use_alt_model)

    def _save_wav(self, audio_data):
        """Сохраняет аудио данные в WAV файл"""
        log(f"💾 Сохраняю в {self.config.audio.WAV_FILE}...")

        with wave.open(self.config.audio.WAV_FILE, "wb") as wf:
            wf.setnchannels(self.config.audio.CHANNELS)
            wf.setsampwidth(self.config.audio.SAMPLE_WIDTH)
            wf.setframerate(self.config.audio.SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())

        log(f"✅ Файл сохранён")

    def _recognize(self, use_alt_model=False):
        """Распознаёт речь из WAV файла"""
        if use_alt_model:
            return self._recognize_whisper()

        log(f"🧠 Загружаю модель {self.config.audio.MODEL_NAME}...")

        try:
            if not self.model:
                self.model = onnx_asr.load_model(self.config.audio.MODEL_NAME)
        except Exception as e:
            log(f"❌ Ошибка загрузки модели: {e}")
            log(f"💡 Модель загрузится автоматически при первом запуске")
            return None

        log("🔍 Распознаю речь...")

        try:
            text = self.model.recognize(self.config.audio.WAV_FILE)
            if text:
                log(f"📝 Распознано: {text}")
            return text
        except Exception as e:
            log(f"❌ Ошибка распознавания: {e}")
            return None

    def _load_whisper_model(self):
        """Загружает Whisper модель (lazy loading)"""
        if self.whisper_pipeline is not None:
            return

        model_id = self.config.audio.ALT_ASR_MODEL
        log(f"🧠 Загружаю Whisper модель {model_id}...")

        self.whisper_model = ORTModelForSpeechSeq2Seq.from_pretrained(
            model_id, subfolder="onnx"
        )
        processor = AutoProcessor.from_pretrained(model_id)

        self.whisper_pipeline = hf_pipeline(
            "automatic-speech-recognition",
            model=self.whisper_model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
        )
        log("✅ Whisper модель загружена")

    def _recognize_whisper(self):
        """Распознаёт речь через Whisper"""
        try:
            self._load_whisper_model()
            result = self.whisper_pipeline(self.config.audio.WAV_FILE)
            text = result["text"]
            if text and text.strip():
                log(f"📝 Распознано (Whisper): {text}")
            else:
                log("⚠️  Whisper вернул пустой текст")
            return text
        except Exception as e:
            log(f"❌ Ошибка Whisper: {e}")
            return None


class PostProcessingService:
    """Сервис для пост-обработки текста с помощью LLM"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.prompt = load_prompt_from_file(
            config.settings.LLM_PROMPT_FILE, "You are a helpful assistant."
        )

    def process(self, text: str) -> str:
        """Отправляет текст в LLM и возвращает обработанный результат"""
        if not self.config.settings.OPENAI_API_KEY:
            log("⚠️  OPENAI_API_KEY не найден. Пост-обработка отключена.")
            return text

        log(
            f"🧠 Отправка текста в LLM (модель: {self.config.settings.OPENAI_MODEL})..."
        )

        for attempt in range(self.config.settings.LLM_MAX_RETRIES):
            try:
                with httpx.Client(
                    timeout=self.config.settings.LLM_TIMEOUT_SEC
                ) as client:
                    response = client.post(
                        f"{self.config.settings.OPENAI_BASE_URL.rstrip('/')}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.config.settings.OPENAI_API_KEY}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": self.config.settings.OPENAI_MODEL,
                            "messages": [
                                {"role": "user", "content": self.prompt},
                                {
                                    "role": "user",
                                    "content": f"<user_input>{text}</user_input>",
                                },
                            ],
                            "temperature": self.config.settings.LLM_TEMPERATURE,
                        },
                    )
                    response.raise_for_status()
                    result = response.json()

                    processed_text = result["choices"][0]["message"]["content"].strip()
                    log(
                        f"✅ LLM ({self.config.settings.OPENAI_MODEL}) обработал: {processed_text}"
                    )
                    return processed_text

            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                log(f"❌ Ошибка при обращении к LLM (попытка {attempt + 1}): {e}")
                if attempt < self.config.settings.LLM_MAX_RETRIES - 1:
                    time.sleep(1)  # Пауза перед повторной попыткой
                continue
            except (KeyError, IndexError) as e:
                log(f"❌ Неожиданный формат ответа от LLM: {e}")
                break  # Не повторяем при ошибках парсинга
            except Exception as e:
                log(f"❌ Неизвестная ошибка при пост-обработке: {e}")
                break  # Не повторяем при других ошибках

        # Резервный вариант: возвращаем исходный текст
        log("⚠️  Не удалось получить ответ от LLM после нескольких попыток.")
        return text


class AsyncTaskRunner:
    """Управляет выполнением асинхронных задач в фоновых потоках"""

    # Режим работы: True для синхронного выполнения (для тестов), False для асинхронного (продакшн)
    _sync_mode = False

    @classmethod
    def run_async(cls, target: Callable, callback: Callable[[any], None]) -> None:
        """
        Запускает задачу в отдельном потоке и возвращает результат в UI-поток

        Args:
            target: Функция для выполнения в фоновом потоке
            callback: Функция для обработки результата в UI-потоке
        """
        if cls._sync_mode:
            # Синхронный режим для тестов - выполняем всё сразу
            result = target()
            callback(result)
        else:
            # Асинхронный режим для продакшна
            def task():
                result = target()
                GLib.idle_add(callback, result)

            thread = threading.Thread(target=task)
            thread.daemon = True
            thread.start()


# ============================================================================
# ИНТЕРФЕЙС
# ============================================================================


class RecognitionWindow:
    """
    Плавающее окно для записи и распознавания речи

    Отвечает только за:
    - Создание и настройку UI элементов
    - Обработку событий UI (клики, drag-and-drop)
    - Синхронизацию состояния UI с состоянием приложения

    Вся бизнес-логика делегируется в ApplicationController
    """

    def __init__(
        self, config: AppConfig, store: Store, monitor_manager: MonitorManager
    ):
        """
        Инициализирует окно с внедрёнными зависимостями

        Args:
            config: Конфигурация приложения
            store: Redux store для управления состоянием
            monitor_manager: Менеджер мониторов
        """
        self.config = config
        self.store = store
        self.monitor_manager = monitor_manager

        self.window = None
        self.button = None
        self.restart_button = None
        self.pp_button = None
        self.app = None

        # Для drag-and-drop (эфемеpное состояние перетаскивания)
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.is_dragging = False
        self.was_moved = False

        # Текущие отступы (кеш для отрисовки и плавного перетаскивания)
        self.current_margin_x = 20
        self.current_margin_y = 50

        # Подписываемся на изменения состояния
        self.store.subscribe(self._render_state)

    @classmethod
    def create_with_defaults(
        cls, factory: ServiceFactory = None
    ) -> "RecognitionWindow":
        """
        Фабричный метод для создания окна с дефолтными зависимостями

        Args:
            factory: Фабрика сервисов для DI (по умолчанию создается с дефолтными реализациями)

        Returns:
            Настроенный экземпляр RecognitionWindow
        """
        config = AppConfig()

        # Создаём фабрику с возможностью инъекции зависимостей
        if factory is None:
            factory = ServiceFactory()

        speech, clipboard, paste, post_processing = factory.create_all_services(config)

        # Создаём один экземпляр MonitorManager для всех нужд
        monitor_manager = MonitorManager(config=config)

        # Путь к файлу настроек
        settings_file = os.path.expanduser(
            "~/.config/float-speech-to-text/settings.json"
        )

        # Загружаем сохранённые настройки
        saved_settings = SettingsPersistenceEffect.load_settings(settings_file)

        # Создаём эффекты (включаем SettingsPersistenceEffect и WindowPersistenceEffect)
        effects = [
            StartRecordingEffect(speech),
            ASREffect(speech, AsyncTaskRunner),
            # LLMEffect(post_processing, AsyncTaskRunner),  # TODO: LLM post-processing disabled - model toggle repurposed for ASR selection
            FinalizeEffect(clipboard, paste, GLib, config),
            RestartEffect(speech, AsyncTaskRunner, AppSettings.RESTART_DELAY_SEC),
            SettingsPersistenceEffect(settings_file),
            WindowPersistenceEffect(
                monitor_manager=monitor_manager,
                window_persistence=WindowPositionPersistence,
                config=config,
            ),
        ]

        # Инициализируем состояние из конфигурации
        initial_state = State(
            llm_enabled=saved_settings.get("llm_enabled", config.settings.LLM_ENABLED),
            auto_paste=saved_settings.get("auto_paste", config.settings.AUTO_PASTE),
            copy_method=saved_settings.get("copy_method", config.settings.COPY_METHOD),
            smart_text_processing=saved_settings.get(
                "smart_text_processing", config.settings.SMART_TEXT_PROCESSING
            ),
            smart_short_phrase_words=saved_settings.get(
                "smart_short_phrase_words", config.settings.SMART_TEXT_SHORT_PHRASE
            ),
        )

        # Создаём хранилище
        store = Store(initial_state, Reducer.reduce, effects)

        return cls(config, store, monitor_manager=monitor_manager)

    def _update_record_button(self, label: str, is_sensitive: bool = True):
        """
        Обновляет состояние кнопки записи (лейбл и чувствительность)

        Args:
            label: Текст лейбла кнопки
            is_sensitive: True если кнопка активна, False если отключена
        """
        if not self.button:
            return

        self.button.set_label(label)
        self.button.set_sensitive(is_sensitive)

    def _update_restart_button(
        self, label: str, is_restart: bool, is_sensitive: bool = True
    ):
        """
        Полностью обновляет состояние кнопки рестарта (лейбл, класс и чувствительность)

        Args:
            label: Текст лейбла кнопки
            is_restart: True для класса restart-button (перезапуск), False для close-button (закрытие)
            is_sensitive: True если кнопка активна, False если отключена
        """
        if not self.restart_button:
            return

        self.restart_button.set_label(label)
        self.restart_button.set_sensitive(is_sensitive)

        # Переключаем CSS класс
        style_context = self.restart_button.get_style_context()
        if is_restart:
            style_context.remove_class("close-button")
            style_context.add_class("restart-button")
        else:
            style_context.remove_class("restart-button")
            style_context.add_class("close-button")

    def _render_state(self, state: State):
        """Redux subscriber - обновляет UI на основе текущего состояния"""
        # 1. Обновляем состояние кнопок на основе фазы
        if state.phase == Phase.IDLE:
            self._update_record_button(self.config.ui.ICON_RECORD, is_sensitive=True)
            self._update_restart_button(
                self.config.ui.ICON_CLOSE, is_restart=False, is_sensitive=True
            )

        elif state.phase == Phase.RECORDING:
            self._update_record_button(self.config.ui.ICON_STOP, is_sensitive=True)
            self._update_restart_button(
                self.config.ui.ICON_RESTART, is_restart=True, is_sensitive=True
            )

        elif state.phase in (Phase.PROCESSING, Phase.POST_PROCESSING):
            self._update_record_button(
                self.config.ui.ICON_PROCESSING, is_sensitive=False
            )
            self._update_restart_button(
                self.config.ui.ICON_CLOSE, is_restart=False, is_sensitive=False
            )

        elif state.phase == Phase.RESTARTING:
            self._update_record_button(
                self.config.ui.ICON_PROCESSING, is_sensitive=False
            )
            self._update_restart_button(
                self.config.ui.ICON_RESTART, is_restart=True, is_sensitive=False
            )

        # 2. Обновляем кнопку PP button
        if self.pp_button:
            icon = (
                self.config.ui.ICON_ALT_MODEL_ON
                if state.llm_enabled
                else self.config.ui.ICON_ALT_MODEL_OFF
            )
            self.pp_button.set_label(icon)
            # Block model toggle during recording/processing
            self.pp_button.set_sensitive(state.phase == Phase.IDLE)

        # 3. Обновляем позицию на основе относительных координат из состояния
        if not self.is_dragging and state.current_monitor_name and self.window:
            monitor = self.monitor_manager.get_monitor_by_name(
                state.current_monitor_name
            )
            if monitor:
                window_width, window_height = self._get_window_size()
                margin_right, margin_top = (
                    self.monitor_manager.calculate_absolute_position(
                        state.rel_x, state.rel_y, window_width, window_height, monitor
                    )
                )

                # Применяем если изменилось
                if (
                    margin_right != self.current_margin_x
                    or margin_top != self.current_margin_y
                ):
                    self.current_margin_x = margin_right
                    self.current_margin_y = margin_top
                    GtkLayerShell.set_margin(
                        self.window, GtkLayerShell.Edge.TOP, int(margin_top)
                    )
                    GtkLayerShell.set_margin(
                        self.window, GtkLayerShell.Edge.RIGHT, int(margin_right)
                    )
                    log(
                        f"📐 Render: Окно позиционировано ({margin_right}, {margin_top}) на {state.current_monitor_name}"
                    )

    def on_button_press(self, _widget, event):
        """Обработчик начала перетаскивания"""
        if event.button == self.config.ui.MOUSE_BUTTON_LEFT:
            self.is_dragging = True
            self.was_moved = False  # Флаг фактического перемещения
            self.drag_start_x = event.x_root
            self.drag_start_y = event.y_root

    def on_button_release(self, _widget, event):
        """Обработчик окончания перетаскивания"""
        if event.button == self.config.ui.MOUSE_BUTTON_LEFT:
            self.is_dragging = False
            # Сохраняем позицию только если окно действительно перемещалось
            if self.was_moved:
                monitor = self.monitor_manager.get_monitor_at_cursor()
                if monitor:
                    monitor_name = self.monitor_manager.get_monitor_identifier(monitor)
                    window_width, window_height = self._get_window_size()

                    # Вычисляем относительную позицию центра окна
                    rel_center_x, rel_center_y = (
                        self.monitor_manager.calculate_relative_position(
                            self.current_margin_x,
                            self.current_margin_y,
                            window_width,
                            window_height,
                            monitor,
                        )
                    )

                    # Диспатчим изменение позиции в Redux -> сохранится через эффект (флаг is_manual=True)
                    self.store.dispatch(
                        WindowPositionChanged(
                            rel_x=rel_center_x, rel_y=rel_center_y, is_manual=True
                        )
                    )

                    # Если монитор изменился (например, перетащили на другой экран), диспатчим это тоже
                    if monitor_name != self.store.state.current_monitor_name:
                        self.store.dispatch(MonitorChanged(monitor_name=monitor_name))
            self.was_moved = False

    def on_motion_notify(self, _widget, event):
        """Обработчик перемещения мыши при перетаскивании"""
        if self.is_dragging:
            # Вычисляем смещение
            dx = event.x_root - self.drag_start_x
            dy = event.y_root - self.drag_start_y

            # Обновляем позицию (кеш)
            # Инвертируем dx, так как окно привязано к правому краю
            self.current_margin_x -= dx
            self.current_margin_y += dy
            self.was_moved = True

            # Плавное обновление через margins (без Redux для производительности драга)
            GtkLayerShell.set_margin(
                self.window, GtkLayerShell.Edge.TOP, int(self.current_margin_y)
            )
            GtkLayerShell.set_margin(
                self.window, GtkLayerShell.Edge.RIGHT, int(self.current_margin_x)
            )

            # Обновляем начальную позицию для следующего движения
            self.drag_start_x = event.x_root
            self.drag_start_y = event.y_root

    def on_restart_clicked(self, button):
        """Обработчик нажатия кнопки перезапуска/закрытия"""
        if self.store.state.phase == Phase.RECORDING:
            # Если идёт запись - диспатчим перезапуск
            self.store.dispatch(UIRestart())
        else:
            # Если не идёт запись - закрываем приложение
            log("🛑 Закрытие приложения...")
            if self.app:
                self.app.quit()

    def on_pp_clicked(self, button):
        """Обработчик нажатия кнопки пост-обработки"""
        # Переключаем состояние пост-обработки через action
        self.store.dispatch(UIToggleAltModel())

        if self.store.state.llm_enabled:
            log("✅ Пост-обработка включена")
        else:
            log("⬜ Пост-обработка выключена")

    def _setup_css_styles(self, screen):
        """Настраивает CSS стили для окна"""
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(self.config.ui.CSS_STYLES)
        Gtk.StyleContext.add_provider_for_screen(
            screen, css_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

    def _setup_transparency(self):
        """Настраивает прозрачность окна"""
        screen = self.window.get_screen()
        visual = screen.get_rgba_visual()
        if visual:
            self.window.set_visual(visual)
        # НЕ устанавливаем set_app_paintable(True) - позволяем GTK рисовать фон с CSS стилями
        return screen

    def _setup_wayland_layer(self):
        """Настраивает Wayland Layer Shell"""
        GtkLayerShell.init_for_window(self.window)

        # Привязываем к верхнему правому углу
        GtkLayerShell.set_anchor(self.window, GtkLayerShell.Edge.TOP, True)
        GtkLayerShell.set_anchor(self.window, GtkLayerShell.Edge.RIGHT, True)

        # Устанавливаем отступы из сохранённой позиции
        GtkLayerShell.set_margin(
            self.window, GtkLayerShell.Edge.TOP, int(self.current_margin_y)
        )
        GtkLayerShell.set_margin(
            self.window, GtkLayerShell.Edge.RIGHT, int(self.current_margin_x)
        )

        # Устанавливаем слой поверх всего
        GtkLayerShell.set_layer(self.window, GtkLayerShell.Layer.OVERLAY)

    def _setup_drag_and_drop(self):
        """Настраивает обработчики для drag-and-drop"""
        self.window.add_events(
            Gdk.EventMask.BUTTON_PRESS_MASK
            | Gdk.EventMask.BUTTON_RELEASE_MASK
            | Gdk.EventMask.POINTER_MOTION_MASK
        )
        self.window.connect("button-press-event", self.on_button_press)
        self.window.connect("button-release-event", self.on_button_release)
        self.window.connect("motion-notify-event", self.on_motion_notify)

    def _create_ui_elements(self, app):
        """Создаёт UI элементы (кнопки)"""
        box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=self.config.ui.BOX_SPACING
        )
        box.set_margin_top(self.config.ui.BOX_MARGIN)
        box.set_margin_bottom(self.config.ui.BOX_MARGIN)
        box.set_margin_start(self.config.ui.BOX_MARGIN)
        box.set_margin_end(self.config.ui.BOX_MARGIN)

        # Кнопка перезапуска записи (изначально показываем закрытие)
        self.restart_button = Gtk.Button(label=self.config.ui.ICON_CLOSE)
        self.restart_button.get_style_context().add_class("close-button")
        self.restart_button.connect("clicked", self.on_restart_clicked)

        # Кнопка записи
        self.button = Gtk.Button(label=self.config.ui.ICON_RECORD)
        self.button.get_style_context().add_class("record-button")
        self.button.connect("clicked", self.on_button_clicked)

        # Кнопка пост-обработки
        initial_pp_icon = (
            self.config.ui.ICON_ALT_MODEL_ON
            if self.config.settings.LLM_ENABLED
            else self.config.ui.ICON_ALT_MODEL_OFF
        )
        self.pp_button = Gtk.Button(label=initial_pp_icon)
        self.pp_button.get_style_context().add_class(
            "autopaste-button"
        )  # Сохраняем старый класс для стилей
        self.pp_button.connect("clicked", self.on_pp_clicked)

        # Сохраняем ссылку на app для возможности закрытия приложения
        self.app = app

        box.add(self.restart_button)
        box.add(self.button)
        box.add(self.pp_button)

        return box

    def on_button_clicked(self, button):
        """Обработчик нажатия кнопки"""
        st = self.store.state
        if st.phase == Phase.IDLE:
            # Начинаем запись через dispatch
            self.store.dispatch(UIStart())
        elif st.phase == Phase.RECORDING:
            # Останавливаем запись и распознаём через dispatch
            self.store.dispatch(UIStop())

    def _get_window_size(self) -> tuple[int, int]:
        """Получает текущие размеры окна, с fallback на preferred size если окно не отрисовано"""
        width = self.window.get_allocated_width()
        height = self.window.get_allocated_height()

        if width <= 1 or height <= 1:
            # Если окно еще не отрисовано или скрыто, запрашиваем желаемый размер
            _min_size, pref_size = self.window.get_preferred_size()
            width = pref_size.width
            height = pref_size.height
            log(f"📐 Окно не отрисовано, используем preferred size: {width}x{height}")

        return width, height

    def on_activate(self, app):
        """Создает и настраивает окно"""
        # Создаем окно
        self.window = Gtk.ApplicationWindow(application=app)

        # Настройка прозрачности
        screen = self._setup_transparency()

        # Настройка Wayland Layer
        self._setup_wayland_layer()

        # CSS стили
        self._setup_css_styles(screen)

        # Перетаскивание (Drag-and-drop)
        self._setup_drag_and_drop()

        # Создаем UI элементы
        box = self._create_ui_elements(app)

        self.window.add(box)
        self.window.show_all()

        # Запускаем мониторинг изменений дисплеев
        display = self.window.get_display()
        self.monitor_manager.start_monitoring(
            display, self._handle_monitor_state_change
        )

        # Запускаем первичную проверку монитора (с использованием внутренней логики ретраев)
        # Это гарантирует, что если монитор не готов при запуске, он будет найден через ретраи.
        self.monitor_manager._handle_monitor_event(display)

        log(f"✅ Окно инициализировано")

    def _handle_monitor_state_change(self, monitor: Optional[Gdk.Monitor]):
        """Обработчик стабильного изменения состояния мониторов"""
        if not monitor:
            log("⚠️  Нет доступных или готовых мониторов. Скрываем окно.")
            self.window.hide()
            return

        monitor_name = self.monitor_manager.get_monitor_identifier(monitor)
        log(f"📺 Монитор для окна: {monitor_name}")

        # Показываем окно если было скрыто
        if not self.window.get_visible():
            self.window.show_all()

        # Загружаем позицию ПЕРЕД диспатчем, чтобы передать её в MonitorChanged
        rel_x, rel_y = WindowPositionPersistence.load_position(monitor_name)
        self.store.dispatch(
            MonitorChanged(monitor_name=monitor_name, rel_x=rel_x, rel_y=rel_y)
        )


def main():
    """Основная функция"""
    log("=" * 50)

    recognition_window = RecognitionWindow.create_with_defaults()
    app = Gtk.Application(application_id=AppConfig.settings.APP_ID)
    app.connect("activate", recognition_window.on_activate)

    # Обработчик Ctrl+C для корректного завершения
    def signal_handler(_sig, _frame):
        log("\n⚠️  Получен сигнал прерывания (Ctrl+C)")
        log("🛑 Останавливаю приложение...")

        # Если идёт запись, диспатчим остановку
        if recognition_window.store.state.phase == Phase.RECORDING:
            recognition_window.store.dispatch(UIStop())
            # Даем время на обработку
            time.sleep(0.5)

        # Завершаем приложение
        app.quit()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    log("💡 Нажмите Ctrl+C для выхода")

    app.run(None)


if __name__ == "__main__":
    main()
