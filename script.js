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

  // Tab switching for course cards (Pertemuan 4, 5, 6)
  const tabButtons = document.querySelectorAll('.dl-tab-btn')
  const tabPanels = document.querySelectorAll('.dl-tab-panel')

  if (tabButtons.length > 0 && tabPanels.length > 0) {
    function activateTab(tabId) {
      tabButtons.forEach(function (btn) {
        const isSelected = btn.getAttribute('data-tab') === tabId
        btn.setAttribute('aria-selected', isSelected ? 'true' : 'false')
        btn.classList.toggle('active', isSelected)
      })
      tabPanels.forEach(function (panel) {
        const isMatch = panel.id === tabId
        panel.hidden = !isMatch
        panel.classList.toggle('active', isMatch)
      })
    }

    tabButtons.forEach(function (btn) {
      btn.addEventListener('click', function () {
        const tabId = btn.getAttribute('data-tab')
        activateTab(tabId)
        if (history.replaceState) {
          history.replaceState(null, '', '#' + tabId)
        }
      })
    })

    // Activate from URL hash if valid
    const initialHash = window.location.hash.replace('#', '')
    if (initialHash && document.getElementById(initialHash)) {
      activateTab(initialHash)
    }
  }
})()
