    override func openDocument(withContentsOf url: URL,
                               display displayDocument: Bool,
                               completionHandler: @escaping (NSDocument?, Bool, Error?) -> Void) {
        // reuse empty view if foremost
        if let currentDocument = self.currentDocument as? Document, currentDocument.isEmpty,
            self.document(for: url) == nil {
            // close the existing view before reusing
            if let oldId = currentDocument.coreViewIdentifier {
                Events.CloseView(viewIdentifier: oldId).dispatch(currentDocument.dispatcher!)
            }

            Events.NewView(path: url.path).dispatchWithCallback(currentDocument.dispatcher!) { (response) in
                DispatchQueue.main.sync {
                    currentDocument.coreViewIdentifier = response
                    currentDocument.editViewController?.visibleLines = 0..<0
                    currentDocument.fileURL = url
                    self.setIdentifier(response, forDocument: currentDocument)
                    completionHandler(currentDocument, false, nil)
                }
            }
        } else {
            super.openDocument(withContentsOf: url,
                               display: displayDocument,
                               completionHandler: completionHandler)
        }
    }

    override func makeUntitledDocument(ofType typeName: String) throws -> NSDocument {
        let document = try Document(type: typeName)
        setupDocument(document, forUrl: nil)
        return document
    }
