project('funcutils', 'cpp',
        version : '0.1',
        default_options : ['warning_level=3', 'cpp_std=c++20  '],
)

pymod = import('python').find_installation('python3')

py3 = find_program('python3')

incdir_pybind11_rel = run_command(py3,
        [
                '-c',
                'import os; import pybind11; print(os.path.relpath(pybind11.get_include()))'
        ],
        check: true
).stdout().strip()
python3 = dependency('python3', )

pybind11 = declare_dependency(
        include_directories: [incdir_pybind11_rel],
)


pymod.extension_module('_funcutils',
                       sources : 'src/main.cpp',
                       dependencies : [python3, pybind11],
                       link_language : 'cpp',
                       override_options: [
                               'cpp_rtti=true',
                       ]
)


exe = executable('main', 'src/main.cpp', include_directories : include_directories('funcutils'))