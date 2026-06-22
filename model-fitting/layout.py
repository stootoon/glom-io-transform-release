 import os

  MODEL_STRS = {"Diag": "fit_diag", "Free": "ffree"}


  def get_split_mode(config, **kwargs):
      split_mode = "random"
      if config:
          if "split" in config["sampler"] and "mode" in config["sampler"]["split"]:
              split_mode = config["sampler"]["split"]["mode"]
      else:
          split_mode = kwargs["split_mode"] if "split_mode" in kwargs else "random"
      return split_mode


  def build_fit_dir(config=None, root="fits", **kwargs):

      def assert_not_none(val, name):
          assert val is not None, f"Missing '{name}'."

      center = (config["init_args"] if config else kwargs).get("center")
      assert_not_none(center, "center")

      standardization = (config if config else kwargs).get("standardization")
      assert_not_none(standardization, "standardization")

      normalization = (config if config else kwargs).get("normalization")
      assert_not_none(normalization, "normalization")
      if isinstance(normalization, list):
          normalization = "_".join(str(n) for n in normalization)

      new_dir = os.path.join(root, f"center={center}/standardization={standardization}/normalization={normalization}")

      sampler_type = config["sampler"]["type"] if config else kwargs.get("sampler_type")
      assert_not_none(sampler_type, "sampler_type")
      new_dir = f"{new_dir}/sampler={sampler_type}"

      split_mode = get_split_mode(config, **kwargs)
      new_dir = f"{new_dir}/mode={split_mode}"
  
      n_od_train = "max"
      if config:
          if "split" in config["sampler"] and "n_od_train" in config["sampler"]["split"]:
              n_od_train = config["sampler"]["split"]["n_od_train"]
      else:
          n_od_train = kwargs["n_od_train"] if "n_od_train" in kwargs else "max"
      new_dir = f"{new_dir}/n_od_train={n_od_train}"

      name = None
      if config and "name" in config:
          name = config["name"]
      elif "name" in kwargs:
          name = kwargs["name"]
      assert_not_none(name, "name")
      new_dir = f"{new_dir}/{name}"

      return new_dir
