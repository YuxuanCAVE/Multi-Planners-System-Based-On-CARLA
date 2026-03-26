import time
import carla


def main():
    host = "127.0.0.1"
    port = 2000
    timeout = 10.0

    client = carla.Client(host, port)
    client.set_timeout(timeout)

    world = client.get_world()
    carla_map = world.get_map()

    spawn_points = carla_map.get_spawn_points()
    debug = world.debug

    print(f"[INFO] total spawn points: {len(spawn_points)}")

    life_time = 600.0  # 显示时间（秒）

    for i, tf in enumerate(spawn_points):
        loc = tf.location

        # 在空中一点显示编号
        text_loc = carla.Location(
            x=loc.x,
            y=loc.y,
            z=loc.z + 1.5,
        )

        # 画编号
        debug.draw_string(
            text_loc,
            str(i),
            draw_shadow=False,
            color=carla.Color(255, 255, 0),
            life_time=life_time,
            persistent_lines=False,
        )

        # 画一个点（更容易看位置）
        debug.draw_point(
            loc,
            size=0.15,
            color=carla.Color(0, 255, 0),
            life_time=life_time,
        )

        # 画朝向箭头（非常关键！）
        yaw = tf.rotation.yaw
        forward = tf.get_forward_vector()

        arrow_end = carla.Location(
            x=loc.x + forward.x * 2.0,
            y=loc.y + forward.y * 2.0,
            z=loc.z + 0.5,
        )

        debug.draw_arrow(
            loc + carla.Location(z=0.5),
            arrow_end,
            thickness=0.1,
            arrow_size=0.3,
            color=carla.Color(0, 0, 255),
            life_time=life_time,
        )

        print(
            f"[{i}] x={loc.x:.2f}, y={loc.y:.2f}, z={loc.z:.2f}, yaw={yaw:.2f}"
        )

    print("\n[INFO] Spawn points drawn in world. Check CARLA window.")
    print("[INFO] They will disappear after", life_time, "seconds.")

    # 等待一段时间方便观察
    time.sleep(life_time)


if __name__ == "__main__":
    main()