from tqdm import tqdm


def main(debug: bool = False) -> None:
    from src.data.exact.core import Core
    from src.data.exact.resources import metadata

    all_core_specifiers = metadata().core_specifier.unique()
    for core_specifier in tqdm(all_core_specifiers, desc="Downloading cores"):
        core: Core = Core.create_core(core_specifier)
        # downloads rf
        if not core.rf_is_downloaded:
            core.rf


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    main(debug=args.debug)
