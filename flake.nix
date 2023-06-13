{
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-22.11";
  inputs.ruff_270.url = "github:nixos/nixpkgs/7913a0c185438e52c04b2ffff539f9fdb6b89e05";

  outputs = { self, nixpkgs, ruff_270 }:
    let
    pkgs = nixpkgs.legacyPackages.x86_64-linux;
    ruffpkg = ruff_270.legacyPackages.x86_64-linux.ruff;
    mypython = pkgs.python310;
    in with pkgs; {
      devShell.x86_64-linux =
        mkShell { buildInputs = [ ruffpkg ];
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
          numpy
          scipy
          mypy
          coverage
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
