{
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs, ... }:
    let
    pkgs = nixpkgs.legacyPackages.x86_64-linux;
    mypython = pkgs.python313;
    pythonpkgs = pkgs.python313Packages;
    in with pkgs; {
      devShell.x86_64-linux =
        mkShell { buildInputs = [
          pythonpkgs.numpy
          pythonpkgs.matplotlib
          pythonpkgs.scipy
          pythonpkgs.pandas
          pythonpkgs.pyarrow
          gh
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
