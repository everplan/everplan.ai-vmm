file(REMOVE_RECURSE
  "libai-vmm.a"
  "libai-vmm.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/ai-vmm.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
