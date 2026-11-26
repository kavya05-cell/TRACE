from live_traffic import LiveTrafficMonitor


monitor = LiveTrafficMonitor(
    interface=None,
    flow_timeout=5
)

try:

    monitor.start()

except KeyboardInterrupt:

    monitor.stop()