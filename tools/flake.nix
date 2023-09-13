{
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-23.05";
  inputs.ruff_287.url = "github:nixos/nixpkgs/e32a63389adb6ee017f8b344e21c80432bb75c10";

  outputs = { self, nixpkgs, ruff_287, ... }:
    let
    pkgs = nixpkgs.legacyPackages.x86_64-linux;
    ruffpkg = ruff_287.legacyPackages.x86_64-linux.ruff;
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
          pythonpkgs.jupyterlab
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
          LD_LIBRARY_PATH=${stdenv.cc.cc.lib}/lib/
          # Allow the use of wheels.
          SOURCE_DATE_EPOCH=$(date +%s)

          # Augment the dynamic linker path
          # Setup the virtual environment if it doesn't already exist.
          VENV=.venv
          if test ! -d $VENV; then
            python -m venv $VENV
            source ./$VENV/bin/activate

            pip install -e ".[develop,gemdash]"
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
