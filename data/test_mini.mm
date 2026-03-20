$( Minimal test database for tensormm development $)

$c |- wff ( ) -> $.
$v ph ps $.

$( Floating hypotheses: ph and ps are wff $)
wph $f wff ph $.
wps $f wff ps $.

$( Axiom: if ph, then ph implies ph $)
${
  ax-1.1 $e |- ph $.
  ax-1 $a |- ( ph -> ph ) $.
$}

$( Simple theorem using ax-1 $)
${
  th1.1 $e |- ps $.
  th1 $p |- ( ps -> ps ) $= wps th1.1 ax-1 $.
$}
