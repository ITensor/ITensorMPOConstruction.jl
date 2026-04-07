"""
    @time_if output_level min_output_level msg ex

Conditionally time and print the execution of `ex` depending on the current
verbosity level.

If `output_level > min_output_level`, this expands to a timed evaluation using
`@time`, prefixing the printed message with `min_output_level` levels of
two-space indentation followed by `msg`. Otherwise, `ex` is evaluated without
timing output.
"""
macro time_if(output_level, min_output_level, msg, ex)
  quote
    local indented_msg = join("  " for _ in 1:($(esc(min_output_level)))) * $(esc(msg))
    local is_timing = $(esc(output_level)) > $(esc(min_output_level))
    is_timing ? (@time indented_msg $(esc(ex))) : $(esc(ex))
  end
end
