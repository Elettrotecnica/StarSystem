#
### Generates a csv summary from experiment results
#

##
# Parameters 
##

set input_files_pattern "04-test-set/05-results/*-Prediction-*.txt"
set output_file "05-results/PredictionResults.csv"
set separator ";"
set under_baseline_results_p f


##
# Helpers
##

# Outputs a formatted csv line
proc write_as_csv {line} {
  global separator wfd
  set l {}
  foreach value $line {
    set v {}
    foreach token $value {
      if {[string is double $token]} {
	regsub -all {\.} $token "," token
      } ; lappend v $token
    } ; lappend l [join $v " "]
  }
  puts $wfd \"[join $l "\"${separator}\""]\"
}


##
# Parsing
##

set headers_written_p f

set wfd [open $output_file w]

foreach f [glob $input_files_pattern] {
  set rfd [open $f r]
  set n_lines 0
  
  set headers [list]
  set values  [list]
  while {[gets $rfd line] >= 0} {
    set line [string trim $line]
    if {$n_lines in {3 4}} {
      if {[lindex $line 3] eq "NO"} {
        break
      }
    }
    if {$n_lines >= 8 && $n_lines <= 23} {
      if {!$headers_written_p} {
        set header [string range $line 3 20]
        set header [string trim $header]
        set header [string tolower $header]
        regsub -all {\s+} $header {_} header
        lappend headers $header
      }
      
      set value [string range $line 23 end]
      set value [string trim $value]
      lappend values $value
    }
    incr n_lines
  }
  close $rfd
  if {$headers ne "" && !$headers_written_p} {
    write_as_csv $headers
    set headers_written_p t
  }
  if {$values ne ""} {
    write_as_csv $values
  }
}
close $wfd
