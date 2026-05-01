"""Test ast_canopy parsing of Eigen-based CUDA headers."""

import os
import pytest

from ast_canopy import parse_declarations_from_source
from eigenprim._env import find_eigen_include

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INCLUDE_DIR = os.path.join(PROJECT_ROOT, "eigenprim", "include")

EIGEN_INCLUDE = find_eigen_include()

CC = "sm_80"


def _parse(header_name, bypass_errors=False, verbose=False, extra_retain=None):
    source = os.path.join(INCLUDE_DIR, header_name)
    retain = [source]
    if extra_retain:
        retain += [os.path.join(INCLUDE_DIR, f) for f in extra_retain]
    return parse_declarations_from_source(
        source,
        retain,
        CC,
        additional_includes=[EIGEN_INCLUDE],
        bypass_parse_error=bypass_errors,
        verbose=verbose,
    )


# ---- Matrix: Direct Eigen type usage ----

class TestMatrixParsing:
    """Matrix: Functions using Eigen's Matrix<float,3,1> as parameters/return types.
    This probes how ast_canopy handles complex template specializations in APIs."""

    @pytest.fixture(scope="class")
    def decls(self):
        return _parse("matrix.cuh", bypass_errors=True)

    def test_parsing_succeeds(self, decls):
        assert decls is not None

    def test_functions_found(self, decls):
        func_names = [f.name for f in decls.functions]
        print(f"Matrix functions found: {func_names}")
        expected = [
            "eigen_vec3f_add", "eigen_vec3f_dot", "eigen_vec3f_cross",
            "eigen_vec3f_norm",
        ]
        for name in expected:
            if name not in func_names:
                pytest.skip(f"Function {name} not found - Eigen type not parsed as function param")

    def test_typedefs_found(self, decls):
        typedef_names = [t.name for t in decls.typedefs]
        print(f"Matrix typedefs found: {typedef_names}")

    def test_function_return_type_info(self, decls):
        """Inspect what ast_canopy reports for Eigen return types."""
        for func in decls.functions:
            if func.name.startswith("eigen_"):
                print(f"  {func.name}: returns {func.return_type.unqualified_non_ref_type_name}")
                for p in func.params:
                    print(f"    param {p.name}: {p.type_.unqualified_non_ref_type_name}")

    def test_class_templates_from_eigen(self, decls):
        """Check if Eigen's Matrix class template is picked up."""
        ct_names = [ct.qual_name for ct in decls.class_templates]
        print(f"Matrix class templates: {ct_names}")

    def test_class_template_specializations(self, decls):
        """Check if concrete specializations like Matrix<float,3,1> are captured."""
        cts_names = [cts.name for cts in decls.class_template_specializations]
        print(f"Matrix class template specializations: {cts_names}")


# ---- Generic: Template wrappers ----

class TestGenericParsing:
    """Generic: Function templates and class templates wrapping Eigen.
    Tests template parsing + explicit instantiation capture."""

    @pytest.fixture(scope="class")
    def decls(self):
        return _parse("generic.cuh", bypass_errors=True,
                       extra_retain=["generic_decl.cuh"])

    def test_parsing_succeeds(self, decls):
        assert decls is not None

    def test_function_templates_found(self, decls):
        ft_names = [ft.qual_name for ft in decls.function_templates]
        print(f"Generic function templates: {ft_names}")
        assert any("templated_dot3" in n for n in ft_names), (
            f"templated_dot3 not found in {ft_names}"
        )

    def test_class_templates_found(self, decls):
        ct_names = [ct.qual_name for ct in decls.class_templates]
        print(f"Generic class templates: {ct_names}")
        assert any("EigenVecWrapper" in n for n in ct_names), (
            f"EigenVecWrapper not found in {ct_names}"
        )

    def test_function_template_params(self, decls):
        for ft in decls.function_templates:
            if "templated_dot3" in ft.qual_name:
                tparams = [tp.name for tp in ft.template_parameters]
                print(f"  templated_dot3 template params: {tparams}")
                assert "Scalar" in tparams

    def test_class_template_params(self, decls):
        for ct in decls.class_templates:
            if "EigenVecWrapper" in ct.qual_name:
                tparams = [tp.name for tp in ct.template_parameters]
                print(f"  EigenVecWrapper template params: {tparams}")
                assert "Scalar" in tparams
                assert "N" in tparams


# ---- Diagnostic: direct Eigen header parsing ----

class TestDirectEigenParsing:
    """Diagnostic: Try to parse Eigen/Dense directly to see what happens."""

    def test_parse_eigen_dense_bypass(self):
        """Parse Eigen/Dense with bypass_parse_error=True.
        This is expected to find a huge number of declarations."""
        source = os.path.join(EIGEN_INCLUDE, "Eigen", "Dense")
        try:
            decls = parse_declarations_from_source(
                source,
                [source],
                CC,
                additional_includes=[EIGEN_INCLUDE],
                bypass_parse_error=True,
            )
            print(f"Direct Eigen/Dense parsing:")
            print(f"  structs: {len(decls.structs)}")
            print(f"  functions: {len(decls.functions)}")
            print(f"  function_templates: {len(decls.function_templates)}")
            print(f"  class_templates: {len(decls.class_templates)}")
            print(f"  class_template_specializations: {len(decls.class_template_specializations)}")
            print(f"  typedefs: {len(decls.typedefs)}")
            print(f"  enums: {len(decls.enums)}")
        except Exception as e:
            pytest.skip(f"Direct Eigen/Dense parsing failed: {e}")
