import logging
import time
from enum import Enum
from statistics import mean
from typing import Dict, List, Optional, Tuple, Union


class Event(Enum):
    START = "start"
    END = "end"
    USER_SPEECH_END = "user_speech_end"
    TRANSCRIPTION_RECEIVED = "transcription_received"
    LLM_START = "llm_start"
    LLM_TTFB = "llm_ttfb"
    LLM_END = "llm_end"
    TTS_START = "tts_start"
    TTS_TTFB = "tts_ttfb"
    TTS_END = "tts_end"


class Metric(Enum):
    LLM_TOTAL_BYTES = "llm_total_bytes"
    TTS_TOTAL_BYTES = "tts_total_bytes"


TraceType = Dict[Union[Event, Metric], Union[float, List[float]]]


class Tracer:
    def __init__(self):
        self.events: List[Tuple[float, Event]] = []
        self.metrics: List[Tuple[float, Metric, float]] = []

    def start(self, start_time: float = None) -> None:
        self.events.append((start_time or time.time(), Event.START))

    def end(self) -> None:
        self.events.append((time.time(), Event.END))
        self.log_timeline()

    def register_event(self, event: Event, event_time: float = None) -> None:
        self.events.append((event_time or time.time(), event))

    def register_metric(self, metric: Metric, metric_value: float, metric_time: float = None) -> None:
        self.metrics.append((metric_time or time.time(), metric, metric_value))

    def _calculate_average(self, start_event: Event, end_event: Event) -> float:
        latencies = [
            self.current_trace[end_event][i] - self.current_trace[start_event][i]
            for i in range(min(len(self.current_trace[start_event]), len(self.current_trace[end_event])))
            if end_event in self.current_trace and start_event in self.current_trace
        ]
        return mean(latencies) if latencies else 0.0

    def _calculate_throughput(self, metric: Metric, start_event: Event, end_event: Event) -> float:
        try:
            throughputs = [
                self.current_trace[metric][i] / (self.current_trace[end_event][i] - self.current_trace[start_event][i])
                for i in range(
                    min(
                        len(self.current_trace[start_event]),
                        len(self.current_trace[end_event]),
                        len(self.current_trace[metric]),
                    )
                )
                if end_event in self.current_trace
                and start_event in self.current_trace
                and metric in self.current_trace
            ]
        except Exception as e:
            logging.error("Error calculating throughput: %s", e, metric, start_event, end_event)
            return 0.0
        return mean(throughputs) if throughputs else 0.0

    def log_avg_stats(self) -> None:
        stats = {
            "Transcription Latency": self._calculate_average(Event.USER_SPEECH_END, Event.TRANSCRIPTION_RECEIVED),
            "LLM Time to First Byte": self._calculate_average(Event.LLM_START, Event.LLM_TTFB),
            "LLM Total Latency": self._calculate_average(Event.LLM_START, Event.LLM_END),
            "LLM Throughput": self._calculate_throughput(Metric.LLM_TOTAL_BYTES, Event.LLM_START, Event.LLM_END),
            "TTS Total Latency": self._calculate_average(Event.TTS_START, Event.TTS_END),
            "TTS Throughput": self._calculate_throughput(Metric.TTS_TOTAL_BYTES, Event.TTS_START, Event.TTS_END),
            "TTS Time to First Byte": self._calculate_average(Event.TTS_START, Event.TTS_TTFB),
            "Total Speech to Speech Latency": self._calculate_average(Event.USER_SPEECH_END, Event.TTS_END),
        }

        logging.info("=== Average Performance Statistics ===")
        for stat_name, stat_value in stats.items():
            logging.info(f"{stat_name}: {stat_value:.2f} {'seconds' if 'Latency' in stat_name else 'bytes/second'}")
        logging.info("======================================")

    def log_current_stats(self) -> None:
        if self.current_trace is None:
            raise RuntimeError("No trace started")

        stats = {
            "Transcription Latency": self._get_event_diff(Event.USER_SPEECH_END, Event.TRANSCRIPTION_RECEIVED),
            "LLM Time to First Byte": self._get_event_diff(Event.LLM_START, Event.LLM_TTFB),
            "LLM Total Latency": self._get_event_diff(Event.LLM_START, Event.LLM_END),
            "LLM Throughput": self._get_throughput(Metric.LLM_TOTAL_BYTES, Event.LLM_START, Event.LLM_END),
            "TTS Total Latency": self._get_event_diff(Event.TTS_START, Event.TTS_END),
            "TTS Throughput": self._get_throughput(Metric.TTS_TOTAL_BYTES, Event.TTS_START, Event.TTS_END),
            "TTS Time to First Byte": self._get_event_diff(Event.TTS_START, Event.TTS_TTFB),
            "Total Speech to Speech Latency": self._get_event_diff(Event.USER_SPEECH_END, Event.TTS_END),
        }

        logging.info("=== Current Performance Statistics ===")
        for stat_name, stat_value in stats.items():
            if stat_value is not None:
                logging.info(f"{stat_name}: {stat_value:.2f} {'seconds' if 'Latency' in stat_name else 'bytes/second'}")
        logging.info("=======================================")

    def _get_event_diff(self, start_event: Event, end_event: Event) -> Optional[float]:
        if start_event in self.current_trace and end_event in self.current_trace:
            return self.current_trace[end_event][-1] - self.current_trace[start_event][-1]
        return None

    def _get_throughput(self, metric: Metric, start_event: Event, end_event: Event) -> Optional[float]:
        try:
            if all(key in self.current_trace for key in (metric, start_event, end_event)):
                return self.current_trace[metric][-1] / (
                    self.current_trace[end_event][-1] - self.current_trace[start_event][-1]
                )
        except Exception as e:
            logging.error("Error calculating throughput: %s", e, metric, start_event, end_event)
            return None

    def log_timeline(self) -> None:
        if not self.events:
            logging.info("No timeline events recorded.")
            return

        logging.info("=== Timeline ===")
        start_time = self.events[0][0]
        last_time = start_time

        logging.info(f"{'Elapsed':>8} {'Delta':>8} {'Event':<25} {'Value'}")
        logging.info("-" * 50)

        for timestamp, event in self.events:
            elapsed_time = timestamp - start_time
            delta_time = timestamp - last_time
            event_name = event.name if isinstance(event, Event) else event.name

            logging.info(f"{elapsed_time:8.3f}s {delta_time:8.3f}s {event_name:<25}")

            last_time = timestamp

        logging.info("=" * 50)


# Global instance
tracer = Tracer()

# Expose global methods
start = tracer.start
end = tracer.end
register_event = tracer.register_event
register_metric = tracer.register_metric
log_avg_stats = tracer.log_avg_stats
log_current_stats = tracer.log_current_stats
log_timeline = tracer.log_timeline
