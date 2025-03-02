# Project Formatting and Styling

The following conventions shall be used throughout the entire repository:

## 1. Adopt Google C++ style guide

- [Link to style guide](https://google.github.io/styleguide/cppguide.html)
- Install the `C/C++` and `C/C++ Include Guard` VSCode extensions.
- Add the following to your `settings.json`:
  ```yml
  "C_Cpp.clang_format_fallbackStyle": "Google",
  "C_Cpp.clang_format_style": "Google",
  "C_Cpp.clang_format_sortIncludes": false,
  "C_Cpp.default.cppStandard": "c++17",
  "C/C++ Include Guard.Macro Type": "Filepath",
  "C/C++ Include Guard.Path Depth": 1,
  "C/C++ Include Guard.Remove Extension": false,
  ```

````
- You should be able to format your C++ code with the Ctrl-Shift-I shortcut.


## 2. Doxygen settings

- Install the `Doxygen Documentation Generator` VSCode extension.
- Add the following to your `settings.json`:
  ```yml
  "doxdocgen.generic.authorName": "Your Name",
  "doxdocgen.generic.authorEmail": "Your Email",
  "doxdocgen.file.fileOrder": [
      "file",
      "brief",
      "date",
      "copyright"
  ],
  "doxdocgen.file.copyrightTag": [
      "",
      "@copyright Copyright (c) {year} Your Name."
  ],
  "doxdocgen.generic.dateFormat": "DD-MM-YYYY",
````
