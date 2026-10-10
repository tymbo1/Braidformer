from scripts.hardware_report import parse_human_bytes, parse_lspci_graphics, parse_nvidia_query


def test_parse_human_bytes() -> None:
    assert parse_human_bytes("8192 MiB") == 8192 * 1024**2
    assert parse_human_bytes("8 GiB") == 8 * 1024**3
    assert parse_human_bytes("not reported") is None


def test_parse_nvidia_query_keeps_vram_as_measured_data() -> None:
    devices = parse_nvidia_query("NVIDIA Test GPU, 8192 MiB, 580.126.18")

    assert devices[0]["model"] == "NVIDIA Test GPU"
    assert devices[0]["vram"]["total_bytes"] == 8192 * 1024**2
    assert devices[0]["vram"]["measured"] is True


def test_parse_lspci_graphics_extracts_identity_without_guessing_vram() -> None:
    output = """24:00.0 VGA compatible controller: NVIDIA Corporation GA104 [GeForce RTX 3060 Ti GDDR6X] (rev a1)
\tKernel driver in use: nvidia
24:00.1 Audio device: NVIDIA Corporation GA104 High Definition Audio Controller [10de:228b] (rev a1)
\tKernel driver in use: snd_hda_intel
"""

    devices = parse_lspci_graphics(output)

    assert devices == [
        {
            "pci_address": "24:00.0",
            "description": "NVIDIA Corporation GA104 [GeForce RTX 3060 Ti GDDR6X] (rev a1)",
            "model": "GeForce RTX 3060 Ti GDDR6X",
            "driver_in_use": "nvidia",
        }
    ]
