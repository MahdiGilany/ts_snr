def register_configs():
    """
    Registers all the structured configs 
    used with hydra. Hydra structured 
    configs can be used to inherit basic config properties 
    for the given object.

    for example: 
    @dataclass
    class FooConfig: 
        a: int = 1
        b: int = 2

    from hydra.core.config_store import ConfigStore
    cs = ConfigStore.instance()
    cs.store(name="foo_base", node=FooConfig)


    Then in a config file you can do:
    foo.yaml: 

    defaults:
        - foo_base

    a: 3
    """
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()

    # from src.data.exact.splits import SplitsConfig

    # cs.store("splits_config_base", SplitsConfig, group="datamodule/splits")
