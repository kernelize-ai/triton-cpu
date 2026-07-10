from triton.knobs import base_knobs, env_bool, env_int, env_str


class cpu_knobs(base_knobs):
    tile_and_fuse: env_bool = env_bool("TRITON_CPU_ENABLE_TILE_AND_FUSE", False)
    warp_size: env_int = env_int("TRITON_CPU_WARP_SIZE", 1)
    feature_override: env_str = env_str("TRITON_CPU_TARGET_FEATURES", "")
    use_sleef: env_bool = env_bool("USE_SLEEF", False)
    use_nexus: env_bool = env_bool("TRITON_CPU_USE_NEXUS", False)
    nexus_device_id: env_int = env_int("TRITON_CPU_NEXUS_DEVICE", 0)
    libomp_path: env_str = env_str("TRITON_LOCAL_LIBOMP_PATH", "/opt/homebrew/opt/libomp/")
    boost_path: env_str = env_str("TRITON_LOCAL_BOOST_PATH", "/opt/homebrew")


cpu = cpu_knobs()
