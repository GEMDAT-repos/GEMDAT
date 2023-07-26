{
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-23.05";
  inputs.ruff_270.url = "github:nixos/nixpkgs/7913a0c185438e52c04b2ffff539f9fdb6b89e05";
  inputs.ruff_274.url = "github:nixos/nixpkgs/1bc7f069be719c24fef65e01383ca48bf3088027";

  outputs = { self, nixpkgs, ruff_274, ... }:
    let
    pkgs = nixpkgs.legacyPackages.x86_64-linux;
    ruffpkg = ruff_274.legacyPackages.x86_64-linux.ruff;
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

            pip install -e ".[develop]"
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
