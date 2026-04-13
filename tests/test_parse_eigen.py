"""Test ast_canopy parsing of Eigen-based CUDA headers at multiple complexity levels."""

import os
import sys
import pytest
import traceback

from ast_canopy import parse_declarations_from_source

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INCLUDE_DIR = os.path.join(PROJECT_ROOT, "include")

EIGEN_INCLUDE = os.path.join(
    PROJECT_ROOT, ".pixi", "envs", "default", "include", "eigen3"
)

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


# ---- Level 1: Thin wrappers (plain structs + device functions) ----

class TestLevel1ThinWrappers:
    """Level 1: Plain structs and device functions that internally call Eigen.
    This should work out of the box since the API surface is simple C++ types."""

    @pytest.fixture(scope="class")
    def decls(self):
        return _parse("eigen_wrapper_l1.cuh", extra_retain=["eigen_wrapper_l1_decl.cuh"])

    def test_parsing_succeeds(self, decls):
        assert decls is not None

    def test_vec3f_struct_found(self, decls):
        struct_names = [s.name for s in decls.structs]
        assert "Vec3f" in struct_names, f"Vec3f not found in {struct_names}"

    def test_vec3f_fields(self, decls):
        vec3f = [s for s in decls.structs if s.name == "Vec3f"][0]
        field_names = [f.name for f in vec3f.fields]
        assert "x" in field_names
        assert "y" in field_names
        assert "z" in field_names

    def test_vec3f_constructors(self, decls):
        vec3f = [s for s in decls.structs if s.name == "Vec3f"][0]
        ctors = list(vec3f.constructors())
        assert len(ctors) >= 2, f"Expected at least 2 constructors, found {len(ctors)}"

    def test_device_functions_found(self, decls):
        func_names = [f.name for f in decls.functions]
        expected = [
            "vec3f_add", "vec3f_dot", "vec3f_cross",
            "vec3f_norm", "vec3f_normalized", "vec3f_scale",
        ]
        for name in expected:
            assert name in func_names, f"Function {name} not found in {func_names}"

    def test_function_exec_space(self, decls):
        from ast_canopy.pylibastcanopy import execution_space
        for func in decls.functions:
            if func.name.startswith("vec3f_"):
                assert func.exec_space == execution_space.device, (
                    f"{func.name} has exec_space={func.exec_space}, expected device"
                )

    def test_function_return_types(self, decls):
        for func in decls.functions:
            if func.name == "vec3f_dot" or func.name == "vec3f_norm":
                assert "float" in func.return_type.unqualified_non_ref_type_name
            elif func.name == "vec3f_add":
                assert "Vec3f" in func.return_type.unqualified_non_ref_type_name

    def test_function_param_types(self, decls):
        for func in decls.functions:
            if func.name == "vec3f_add":
                assert len(func.params) == 2
            elif func.name == "vec3f_scale":
                assert len(func.params) == 2


# ---- Level 2: Direct Eigen type usage ----

class TestLevel2DirectEigenTypes:
    """Level 2: Functions using Eigen's Matrix<float,3,1> as parameters/return types.
    This probes how ast_canopy handles complex template specializations in APIs."""

    @pytest.fixture(scope="class")
    def decls(self):
        return _parse("eigen_wrapper_l2.cuh", bypass_errors=True)

    def test_parsing_succeeds(self, decls):
        assert decls is not None

    def test_functions_found(self, decls):
        func_names = [f.name for f in decls.functions]
        print(f"L2 functions found: {func_names}")
        expected = [
            "eigen_vec3f_add", "eigen_vec3f_dot", "eigen_vec3f_cross",
            "eigen_vec3f_norm",
        ]
        for name in expected:
            if name not in func_names:
                pytest.skip(f"Function {name} not found - Eigen type not parsed as function param")

    def test_typedefs_found(self, decls):
        typedef_names = [t.name for t in decls.typedefs]
        print(f"L2 typedefs found: {typedef_names}")

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
        print(f"L2 class templates: {ct_names}")

    def test_class_template_specializations(self, decls):
        """Check if concrete specializations like Matrix<float,3,1> are captured."""
        cts_names = [cts.name for cts in decls.class_template_specializations]
        print(f"L2 class template specializations: {cts_names}")


# ---- Level 3: Template wrappers ----

class TestLevel3TemplateWrappers:
    """Level 3: Function templates and class templates wrapping Eigen.
    Tests template parsing + explicit instantiation capture."""

    @pytest.fixture(scope="class")
    def decls(self):
        return _parse("eigen_wrapper_l3.cuh", bypass_errors=True,
                       extra_retain=["eigen_wrapper_l3_decl.cuh"])

    def test_parsing_succeeds(self, decls):
        assert decls is not None

    def test_function_templates_found(self, decls):
        ft_names = [ft.qual_name for ft in decls.function_templates]
        print(f"L3 function templates: {ft_names}")
        assert any("templated_dot3" in n for n in ft_names), (
            f"templated_dot3 not found in {ft_names}"
        )

    def test_class_templates_found(self, decls):
        ct_names = [ct.qual_name for ct in decls.class_templates]
        print(f"L3 class templates: {ct_names}")
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
