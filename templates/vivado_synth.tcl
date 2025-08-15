# vivado_synth.tcl — HLS OOC synth that builds HLS sub-IPs too

set tcldir [file dirname [info script]]
source [file join $tcldir project.tcl]    ;# project_name, part, ...

set sol_dir "${project_name}_prj/solution1"
set workdir "ooc_${project_name}_synth"
file mkdir $workdir

create_project ooc $workdir -part $part -force
set_property target_language Verilog [current_project]

# ───────────────────── 1) Create sub-IPs from *_ip.tcl ─────────────────────
set ip_tcls [concat \
  [glob -nocomplain -directory $sol_dir/syn/verilog *_ip.tcl] \
  [glob -nocomplain -directory $sol_dir/impl/ip      *_ip.tcl] \
]
if {[llength $ip_tcls] > 0} {
  puts "INFO: Sourcing HLS IP Tcl scripts:"
  foreach t $ip_tcls {
    puts "  - $t"
    set old [pwd]
    cd [file dirname $t]
    # many HLS scripts assume they run in their own dir
    source [file tail $t]
    cd $old
  }
  if {[llength [get_ips -quiet *]]} {
    upgrade_ip -quiet [get_ips *]
    generate_target -force {synthesis} [get_ips *]
  }
} else {
  puts "INFO: No *_ip.tcl files found (nothing to create)."
}

# ───────────────────── 2) Read packaged IP (.xci/.xcix) ────────────────────
set ip_files [concat \
  [glob -nocomplain -directory $sol_dir/impl/ip           *.xci *.xcix] \
  [glob -nocomplain -directory $sol_dir/syn/vivado_ip/ip  *.xci *.xcix] \
]
if {[llength $ip_files] > 0} {
  puts "INFO: Reading packaged IP:"
  foreach f $ip_files { puts "  - $f" }
  read_ip $ip_files
  upgrade_ip -quiet [get_ips *]
  generate_target -force {synthesis} [get_ips *]
}

# ───────────────────── 3) Read RTL (syn + impl/ip/hdl) ─────────────────────
set rtl_files {}
foreach d [list $sol_dir/syn/verilog $sol_dir/impl/ip/hdl] {
  if {[file isdirectory $d]} {
    foreach f [glob -nocomplain -directory $d *.v *.sv *.vh] {
      lappend rtl_files $f
    }
  }
}
if {[llength $rtl_files] == 0} {
  error "No RTL sources found under: $sol_dir/syn/verilog and $sol_dir/impl/ip/hdl"
}
read_verilog -sv $rtl_files

# ───────────────────── 4) Auto-detect top safely ───────────────────────────
proc _has_module {fname mname} {
  if {[catch {set fh [open $fname r]}]} { return 0 }
  set txt [read $fh]; close $fh
  # Build the regexp with braces so [] aren’t Tcl command substitution,
  # then inject the module name via 'format'.
  set pat [format {module[\ \t]+%s[\ \t]*\(} $mname]
  return [regexp -lineanchor -- $pat $txt]
}
set candidates [list ${project_name}_stream ${project_name}_axi ${project_name}]
set top_name ""
foreach cand $candidates {
  set found 0
  foreach f $rtl_files { if {[_has_module $f $cand]} { set found 1; break } }
  if {$found} { set top_name $cand; break }
}
if {$top_name eq ""} {
  puts "WARN: Could not auto-detect top; defaulting to ${project_name}"
  set top_name ${project_name}
}
puts "INFO: Using TOP = $top_name"

# ───────────────────── 5) Synthesize & report ──────────────────────────────
# Reports dir inside the Vivado project folder
set rpt_dir [file normalize [file join $workdir reports]]
file mkdir $rpt_dir

synth_design -top $top_name -part $part -mode out_of_context

report_ip_status                          -file [file join $rpt_dir vivado_ip_status.rpt]
report_utilization -hierarchical          -file [file join $rpt_dir vivado_synth.rpt]
report_timing_summary -no_detailed_paths  -file [file join $rpt_dir vivado_timing.rpt]
report_clock_utilization                  -file [file join $rpt_dir vivado_clock_util.rpt]
report_control_sets                       -file [file join $rpt_dir vivado_control_sets.rpt]