import atexit
import queue

import numpy as np
import pyaudio
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Int16MultiArray, MultiArrayDimension


class AudioListenerNode(Node):
    def __init__(self, node_name: str) -> None:
        super().__init__(node_name)

        self.declare_parameters(
            namespace="",
            parameters=[
                ("channels", 1),
                ("frames_per_buffer", 4000),
                ("rate", 16000),
            ],
        )

        self.channels_ = self.get_parameter("channels").get_parameter_value().integer_value
        self.frames_per_buffer_ = self.get_parameter("frames_per_buffer").get_parameter_value().integer_value
        self.rate_ = self.get_parameter("rate").get_parameter_value().integer_value

        self.get_logger().info(f"Starte AudioListener mit {self.channels_} Kanal/Kanälen, "
                               f"Puffer={self.frames_per_buffer_}, Rate={self.rate_} Hz")

        self._audio_queue = queue.Queue()

        self.audio_publisher_ = self.create_publisher(
            Int16MultiArray, "~/audio", qos_profile_sensor_data
        )

        self.pyaudio_ = pyaudio.PyAudio()

        def pyaudio_callback(in_data, frame_count, time_info, status):
            self._audio_queue.put(in_data)
            return (None, pyaudio.paContinue)

        self.stream_ = self.pyaudio_.open(
            format=pyaudio.paInt16,
            channels=self.channels_,
            rate=self.rate_,
            input=True,
            frames_per_buffer=self.frames_per_buffer_,
            input_device_index=0,
            stream_callback=pyaudio_callback,
            start=False
        )

        self.stream_.start_stream()

        self._timer = self.create_timer(
            0.05,  # 50 ms
            self.publish_audio
        )

        atexit.register(self.cleanup_)

    def publish_audio(self):
        while not self._audio_queue.empty():
            raw_data = self._audio_queue.get()
            try:
                audio_samples = np.frombuffer(raw_data, dtype=np.int16)
                msg = Int16MultiArray()
                msg.data = audio_samples.tolist()
                msg.layout.data_offset = 0
                msg.layout.dim.append(MultiArrayDimension(
                    label="audio", size=len(audio_samples), stride=1
                ))
                self.audio_publisher_.publish(msg)
            except Exception as e:
                self.get_logger().error(f"Fehler beim Verarbeiten von Audio: {e}")

    def cleanup_(self):
        if self.stream_.is_active():
            self.stream_.stop_stream()
        self.stream_.close()
        self.pyaudio_.terminate()
        self.get_logger().info("PyAudio geschlossen.")


def main(args=None):
    rclpy.init(args=args)
    node = AudioListenerNode("audio_listener")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()