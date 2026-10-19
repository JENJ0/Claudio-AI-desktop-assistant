"""List audio devices: copy the index or part of the name into config/claudio.toml."""

import sounddevice as sd


def main() -> None:
    apis = sd.query_hostapis()
    default_in, default_out = sd.default.device
    print(f"{'#':>3}  {'in':>2} {'out':>3}  {'API':<22} nome")
    for i, dev in enumerate(sd.query_devices()):
        mark = " <- predefinito" if i in (default_in, default_out) else ""
        api = apis[dev["hostapi"]]["name"]
        print(
            f"{i:>3}  {dev['max_input_channels']:>2} {dev['max_output_channels']:>3}  "
            f"{api:<22} {dev['name']}{mark}"
        )
    print("\nConsiglio: usa un dispositivo 'Windows WASAPI' per la latenza più bassa.")


if __name__ == "__main__":
    main()
