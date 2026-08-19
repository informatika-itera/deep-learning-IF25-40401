// Deep Learning RPS — lightweight static-page behavior
(function () {
  'use strict'

  // Keep the "Kembali ke Atas" link smooth even when the page is opened locally.
  document.querySelectorAll('a[href="#top"]').forEach(function (link) {
    link.addEventListener('click', function (event) {
      event.preventDefault()
      window.scrollTo({ top: 0, behavior: 'smooth' })
    })
  })

  // Add a small table affordance on narrow screens where wide cells scroll.
  document.querySelectorAll('.rps-grade-table').forEach(function (table) {
    table.setAttribute('tabindex', '0')
    table.setAttribute('aria-label', 'Tabel yang dapat digeser secara horizontal')
  })
})()
