import cv2


class ComparisonVisualizer:

    @staticmethod
    def show(
        file_name,
        image,
        pipeline_pressure,
        mobile_pressure,
        mobile_scale,
        mobile_latency
    ):

        vis = image.copy()

        if pipeline_pressure is None:
            pipeline_text = "Pipeline: ERROR"
        else:
            pipeline_text = (
                f"Pipeline: {pipeline_pressure:.2f} MPa"
            )

        cv2.putText(
            vis,
            pipeline_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            vis,
            f"MobileNet: {mobile_pressure:.2f} MPa",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.putText(
            vis,
            f"Scale: {mobile_scale:.1f}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.putText(
            vis,
            f"Latency: {mobile_latency:.1f} ms",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        if pipeline_pressure is not None:

            delta = abs(
                pipeline_pressure -
                mobile_pressure
            )

            cv2.putText(
                vis,
                f"Delta: {delta:.2f}",
                (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.imshow(
            f"Comparison: {file_name}",
            vis
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()