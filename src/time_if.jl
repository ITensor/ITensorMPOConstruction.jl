macro time_if(output_level, min_output_level, msg, ex)
  quote
    local indented_msg = join("  " for _ in 1:$(esc(min_output_level))) * $(esc(msg))
    local is_timing = $(esc(output_level)) > $(esc(min_output_level))
    is_timing ? (@time indented_msg $(esc(ex))) : $(esc(ex))
  end
end
