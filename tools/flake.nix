{
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  inputs.ruff_287.url = "github:nixos/nixpkgs/e32a63389adb6ee017f8b344e21c80432bb75c10";
  inputs.ruff_0_1_5.url = "github:nixos/nixpkgs/aa1d7f6320c32010a990ba6c78fcb24cf9e99270";

  outputs = { self, nixpkgs, ruff_0_1_5, ... }:
    let
    pkgs = nixpkgs.legacyPackages.x86_64-linux;
    ruffpkg = ruff_0_1_5.legacyPackages.x86_64-linux.ruff;
    mypython = pkgs.python311;
    pythonpkgs = pkgs.python311Packages;
    in with pkgs; {
      devShell.x86_64-linux =
        mkShell { buildInputs = [
          ruffpkg
          pythonpkgs.numpy
          pythonpkgs.matplotlib
          pythonpkgs.scipy
          pythonpkgs.pandas
          pythonpkgs.pyarrow
        ];
        pythonWithPkgs = mypython.withPackages (pythonPkgs: with pythonPkgs; [
          # This list contains tools for Python development.
          # You can also add other tools, like black.
          #
          # Note that even if you add Python packages here like PyTorch or Tensorflow,
          # they will be reinstalled when running `pip -r requirements.txt` because
          # virtualenv is used below in the shellHook.
          pip
          setuptools
          virtualenvwrapper
          wheel
        ]);
        shellHook = ''
          # fixes libstdc++ issues and libgl.so issues
          export LD_LIBRARY_PATH=${stdenv.cc.cc.lib}/lib/
          # Allow the use of wheels.
          export SOURCE_DATE_EPOCH=$(date +%s)

          # Augment the dynamic linker path
          # Setup the virtual environment if it doesn't already exist.
          VENV=.venv
          if test ! -d $VENV; then
            python -m venv $VENV
            source ./$VENV/bin/activate

            pip install -e ".[develop,gemdash,docs]"
          fi

          export PYTHONPATH=`pwd`/$VENV/${mypython.sitePackages}/:$PYTHONPATH
          source ./$VENV/bin/activate
          if test -f "source_me.sh"; then
            source source_me.sh
          fi
        '';
        };
   };
}
