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
            currentDocument.coreViewIdentifier = nil;

            Events.NewView(path: url.path).dispatchWithCallback(currentDocument.dispatcher!) { (response) in
                DispatchQueue.main.sync {

                    switch response {
                    case .ok(let result):
                        let result = result as! String
                        currentDocument.coreViewIdentifier = result
                        currentDocument.fileURL = url
                        self.setIdentifier(result, forDocument: currentDocument)
                        currentDocument.editViewController!.redrawEverything()
                        completionHandler(currentDocument, false, nil)

                    case .error(let error):
                        let userInfo = [NSLocalizedDescriptionKey: error.message]
                        let nsError = NSError(domain: "xi.io.error", code: error.code,
                                              userInfo: userInfo)
                        completionHandler(nil, false, nsError)
                        currentDocument.close()
                    }
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
