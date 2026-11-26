import time, threading
from collections import Counter, deque

try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP
except ImportError:
    sniff = IP = TCP = UDP = ICMP = None

class LiveTrafficMonitor:
    def __init__(self, interface=None, window_seconds=5, max_events=1000):
        self.interface = interface
        self.window_seconds = window_seconds
        self.lock = threading.Lock()
        self.running = False
        self.packet_count = 0
        self.flow_count = 0
        self.flows = {}
        self.events = deque(maxlen=max_events)
        self.alerts = deque(maxlen=500)
        self.last_window = time.time()
        self.protocols = Counter()
        self.src_counts = Counter()

    @staticmethod
    def _key(pkt):
        if IP not in pkt:
            return None

        proto = "TCP" if TCP in pkt else "UDP" if UDP in pkt else "ICMP" if ICMP in pkt else "OTHER"

        sport = int(pkt[TCP].sport if TCP in pkt else pkt[UDP].sport if UDP in pkt else 0)
        dport = int(pkt[TCP].dport if TCP in pkt else pkt[UDP].dport if UDP in pkt else 0)

        return (pkt[IP].src, pkt[IP].dst, sport, dport, proto)

    def process_packet(self, pkt):
        key = self._key(pkt)
        if key is None:
            return

        now = time.time()
        size = len(pkt)
        flags = str(pkt[TCP].flags) if TCP in pkt else ""

        with self.lock:
            self.packet_count += 1
            self.protocols[key[4]] += 1
            self.src_counts[key[0]] += 1

            flow = self.flows.setdefault(key, {
                "first": now,
                "last": now,
                "packets": 0,
                "bytes": 0,
                "syn": 0,
                "rst": 0,
                "fin": 0
            })

            flow["last"] = now
            flow["packets"] += 1
            flow["bytes"] += size
            flow["syn"] += int("S" in flags and "A" not in flags)
            flow["rst"] += int("R" in flags)
            flow["fin"] += int("F" in flags)

            self.flow_count = len(self.flows)

            self.events.append({
                "time": now,
                "src": key[0],
                "dst": key[1],
                "sport": key[2],
                "dport": key[3],
                "proto": key[4],
                "bytes": size,
                "flags": flags
            })

            if now - self.last_window >= self.window_seconds:
                self._evaluate_window(now)
                self.last_window = now

    def _evaluate_window(self, now):
        total = sum(f["packets"] for f in self.flows.values())
        syn = sum(f["syn"] for f in self.flows.values())
        unique_ports = len({k[3] for k in self.flows})
        top_source, top_count = self.src_counts.most_common(1)[0] if self.src_counts else ("-", 0)

        reasons = []
        score = 0

        if syn >= 20 and syn / max(total, 1) > .65:
            reasons.append("High SYN-only traffic ratio")
            score += 45

        if unique_ports >= 20:
            reasons.append(f"Port scan pattern ({unique_ports} destination ports)")
            score += 35

        if top_count >= 100:
            reasons.append(f"High traffic volume from {top_source}")
            score += 25

        if len(self.flows) >= 250:
            reasons.append("High concurrent flow count")
            score += 20

        score = min(score, 100)

        if score >= 40:
            self.alerts.appendleft({
                "time": now,
                "severity": "HIGH" if score >= 70 else "MEDIUM",
                "score": score,
                "src": top_source,
                "description": "; ".join(reasons),
                "flows": len(self.flows),
                "packets": total
            })

        self.flows.clear()
        self.src_counts.clear()

    def snapshot(self):
        with self.lock:
            return {
                "packet_count": self.packet_count,
                "flow_count": self.flow_count,
                "protocols": dict(self.protocols),
                "events": list(self.events)[-100:],
                "alerts": list(self.alerts),
                "flows": dict(self.flows)
            }

    def start(self):
        if sniff is None:
            raise RuntimeError("Scapy not installed. Run: pip install scapy")

        if self.running:
            return

        self.running = True
        threading.Thread(target=self._capture, daemon=True).start()

    def _capture(self):
        try:
            sniff(
                iface=self.interface,
                prn=self.process_packet,
                store=False,
                stop_filter=lambda _: not self.running
            )
        except Exception as e:
            with self.lock:
                self.alerts.appendleft({
                    "time": time.time(),
                    "severity": "ERROR",
                    "score": 0,
                    "src": "-",
                    "description": str(e),
                    "flows": 0,
                    "packets": 0
                })
        finally:
            self.running = False

    def stop(self):
        self.running = False